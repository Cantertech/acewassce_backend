import os
import json
from typing import TypedDict, List, Optional, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from database import get_db

load_dotenv()

# --- 1. THE STATE ---
class GradingState(TypedDict):
    """
    Represents the state of the batch grading workflow.
    """
    attempt_id: str
    submissions: List[dict] # All images from the frontend
    routed_work: Dict[str, List[str]] # { "4": ["url1", "url2"], "5": ["url3"] }
    grading_results: List[dict] # [ { question_number: 4, score: 7, feedback: "..." } ]
    total_score: int

# --- 2. THE NODES ---

async def router_node(state: GradingState):
    """
    Identifies which question each image belongs to by reading handwriting.
    """
    print(f"--- [NODE: Router] Analyzing {len(state['submissions'])} images for question numbers ---")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    routed = {}
    
    for sub in state['submissions']:
        image_url = sub['image_url']
        
        system_prompt = (
            "You are an image router. Look at this student's handwritten math paper. "
            "Identify the question number written on top of the page. "
            "Only return the number (e.g. '4', '5', '12'). If no number is clear, return 'unknown'."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        
        try:
            response = await llm.ainvoke([message])
            q_num = response.content.strip().lower().replace("question", "").replace("q", "").strip()
            
            print(f"DEBUG [Router]: Image identified as Question {q_num}")
            
            if q_num not in routed:
                routed[q_num] = []
            routed[q_num].append(image_url)
        except Exception as e:
            print(f"Routing error for image: {e}")
            
    state["routed_work"] = routed
    print(f"--- [ROUTING COMPLETE] Groups: {list(routed.keys())} ---")
    return state

async def batch_grade_node(state: GradingState):
    """
    Grades each identified question group against the marking scheme.
    """
    print(f"--- [NODE: Batch Grade] Starting evaluation for {len(state['routed_work'])} questions ---")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    db = get_db()
    
    results = []
    total_score = 0
    
    # 1. Get Exam ID for this attempt to fetch theory questions
    attempt_res = db.table("exam_attempts").select("exam_id").eq("id", state["attempt_id"]).single().execute()
    exam_id = attempt_res.data["exam_id"]
    
    # 2. Fetch all theory questions for this exam
    q_res = db.table("questions").select("*").eq("exam_id", exam_id).eq("is_mcq", False).execute()
    questions_map = {str(q["question_number"]): q for q in q_res.data}

    for q_num, urls in state["routed_work"].items():
        if q_num not in questions_map:
            print(f"WARNING: Q{q_num} skipped (No rubric found in database).")
            continue
            
        question = questions_map[q_num]
        print(f"\n--- [FORENSIC GRADING] Question {q_num} ---")
        
        # Robustly get marking scheme
        rubric = question.get('marking_scheme') or question.get('rubric') or question.get('marking_guide') or "Grade based on standard WAEC marking criteria."
        print(f"DEBUG: Using Rubric (first 100 chars): {rubric[:100]}...")
        
        # Combine all images for this question into one evaluation
        eval_prompt = (
            f"You are a strict WAEC Examiner. Evaluate the student's handwritten workings against the rubric.\n\n"
            f"RUBRIC for Q{q_num}:\n{rubric}\n\n"
            "STUDENT WORKINGS (Images attached below):\n"
            "Identify what the student wrote, compare it to the marking guide, and award marks for method and accuracy."
        )
        
        messages = [
            SystemMessage(content="Output JSON only: { 'score': int, 'feedback': 'string', 'ocr_transcript': 'string' }"),
            HumanMessage(content=[{"type": "text", "text": eval_prompt}])
        ]
        
        for url in urls:
            messages[1].content.append({"type": "image_url", "image_url": {"url": url}})
            
        try:
            response = await llm.ainvoke(messages)
            content = response.content.replace("```json", "").replace("```", "").strip()
            grading = json.loads(content)
            
            score = grading.get("score", 0)
            feedback = grading.get("feedback", "No feedback.")
            ocr = grading.get("ocr_transcript", "Not extracted.")
            
            print(f"AI EXTRACTED: \"{ocr[:200]}...\"")
            print(f"AI REASONING: {feedback}")
            print(f"RESULT: {score} marks awarded.\n")
            
            total_score += score
            
            # Save detail record
            detail_data = {
                "attempt_id": state["attempt_id"],
                "question_number": int(q_num),
                "marks_attained": score,
                "max_marks": question.get("points", 10),
                "feedback": f"[OCR: {ocr}] {feedback}",
                "image_url": urls[0] 
            }
            db.table("theory_submissions").insert(detail_data).execute()
            
            results.append({
                "question_number": q_num,
                "score": score,
                "feedback": feedback
            })
        except Exception as e:
            print(f"!!! CRITICAL GRADING FAILURE for Q{q_num}: {e}")

    state["grading_results"] = results
    state["total_score"] = total_score
    return state

# --- 3. THE GRAPH COMPILATION ---

workflow = StateGraph(GradingState)

workflow.add_node("router", router_node)
workflow.add_node("batch_grade", batch_grade_node)

workflow.add_edge(START, "router")
workflow.add_edge("router", "batch_grade")
workflow.add_edge("batch_grade", END)

ace_wassce_grader = workflow.compile()

async def run_grader(attempt_id: str, submissions: List[dict]):
    """
    Wrapper function to invoke the batch grading graph.
    """
    initial_state = {
        "attempt_id": attempt_id,
        "submissions": submissions,
        "routed_work": {},
        "grading_results": [],
        "total_score": 0
    }
    
    print(f"🚀 Launching Batch AI Grader for attempt: {attempt_id}")
    final_output = await ace_wassce_grader.ainvoke(initial_state)
    return final_output
