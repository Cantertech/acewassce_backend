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
    Identifies ALL question numbers present on each image.
    """
    print(f"--- [NODE: Router] Identifying ALL questions in {len(state['submissions'])} images ---")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    routed = {}
    
    for sub in state['submissions']:
        image_url = sub['image_url']
        
        system_prompt = (
            "You are a document scanner. Look at this student's handwritten paper. "
            "Identify EVERY question number present on this page. Students often write multiple answers (e.g., 6, 7, and 8) on one sheet. "
            "Return a comma-separated list of numbers only (e.g. '6, 7, 8'). If none are found, return 'unknown'."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        
        try:
            response = await llm.ainvoke([message])
            raw_nums = response.content.strip().lower().replace("questions", "").replace("question", "").replace("q", "").strip()
            
            # Extract list of numbers
            q_nums = [n.strip() for n in raw_nums.split(",") if n.strip().isdigit()]
            
            print(f"DEBUG [Router]: Image contains Questions: {q_nums if q_nums else 'None'}")
            
            for q_num in q_nums:
                if q_num not in routed:
                    routed[q_num] = []
                routed[q_num].append(image_url)
        except Exception as e:
            print(f"Routing error for image: {e}")
            
    state["routed_work"] = routed
    print(f"--- [ROUTING COMPLETE] Master Map: {list(routed.keys())} ---")
    return state

async def batch_grade_node(state: GradingState):
    """
    Grades each identified question group against the marking scheme with strict adherence.
    """
    print(f"--- [NODE: Batch Grade] Commencing strict rubric-based evaluation ---")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    db = get_db()
    
    results = []
    total_score = 0
    
    # 1. Get Exam ID
    attempt_res = db.table("exam_attempts").select("exam_id").eq("id", state["attempt_id"]).single().execute()
    exam_id = attempt_res.data["exam_id"]
    
    # 2. Fetch all theory questions
    q_res = db.table("questions").select("*").eq("exam_id", exam_id).eq("is_mcq", False).execute()
    questions_map = {str(q["question_number"]): q for q in q_res.data}

    for q_num, urls in state["routed_work"].items():
        if q_num not in questions_map:
            print(f"WARNING: Q{q_num} skipped (No marking scheme found in database).")
            continue
            
        # Rate limit protection: Wait a bit before each question
        import asyncio
        await asyncio.sleep(1.5)
            
        question = questions_map[q_num]
        print(f"\n--- [STRICT MARKING] Question {q_num} (Max Marks: {question.get('points', 10)}) ---")
        
        # Get rubric from DB (this should contain your JSON data)
        rubric = question.get('marking_scheme') or question.get('rubric') or question.get('marking_guide')
        
        if not rubric:
            print(f"CRITICAL: No specific rubric for Q{q_num}. AI is NOT allowed to guess.")
            rubric = "ERROR: No marking scheme provided. Award 0 marks and state 'Missing Rubric' in feedback."

        eval_prompt = (
            f"You are a Senior WAEC Examiner. Your task is to award marks SOLELY based on the provided OFFICIAL MARKING SCHEME.\n\n"
            f"OFFICIAL MARKING SCHEME for Q{q_num}:\n{rubric}\n\n"
            "STUDENT WORKINGS (Images attached):\n"
            "1. Transcribe the student's work for this specific question.\n"
            "2. Match each step of the student's work to the official marking scheme steps.\n"
            "3. If the work does not match the scheme, award 0 for that step.\n"
            "4. Provide a summative reasoning for the total marks awarded based strictly on the scheme."
        )
        
        messages = [
            SystemMessage(content="Output JSON only: { 'score': int, 'summative_reasoning': 'string', 'ocr_transcript': 'string' }"),
            HumanMessage(content=[{"type": "text", "text": eval_prompt}])
        ]
        
        for url in urls:
            messages[1].content.append({"type": "image_url", "image_url": {"url": url}})
            
        try:
            response = await llm.ainvoke(messages)
            content = response.content.replace("```json", "").replace("```", "").strip()
            grading = json.loads(content)
            
            score = grading.get("score", 0)
            reasoning = grading.get("summative_reasoning", "No reasoning provided.")
            ocr = grading.get("ocr_transcript", "Not extracted.")
            
            print(f"EXTRACTED WORK: \"{ocr[:300]}...\"")
            print(f"RUBRIC COMPLIANCE: {reasoning}")
            print(f"FINAL MARK: {score}/{question.get('points', 10)}")
            
            total_score += score
            
            detail_data = {
                "attempt_id": state["attempt_id"],
                "question_number": int(q_num),
                "marks_attained": score,
                "max_marks": question.get("points", 10),
                "feedback": f"[SUMMARIZED REASONING]: {reasoning} | [TRANSCRIPT]: {ocr}",
                "image_url": urls[0] 
            }
            db.table("theory_submissions").insert(detail_data).execute()
            
            results.append({"question_number": q_num, "score": score, "reasoning": reasoning})
        except Exception as e:
            print(f"!!! FAILURE GRADING Q{q_num}: {e}")

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
