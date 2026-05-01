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
        try:
            image_url = sub['image_url']
            manual_tags = sub.get("feedback") # We store tags here
            
            q_nums = []
            
            if manual_tags and manual_tags.strip().lower() != "unknown":
                # Use Manual Tags provided by the student
                import re
                # Split by comma or space and clean up
                raw_parts = re.split(r'[,|\s]', manual_tags)
                # Normalize: Strip letters (e.g., 9a -> 9) to match DB question numbers
                q_nums = [re.sub(r'[^0-9]', '', p.strip()) for p in raw_parts if p.strip()]
                # Filter out empty results after regex
                q_nums = [n for n in q_nums if n]
                
                print(f"DEBUG [Router]: Using NORMALIZED MANUAL TAGS: {q_nums}")
            else:
                # Fallback to AI Identification
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
                    q_nums = [n.strip() for n in raw_nums.split(",") if n.strip()]
                    print(f"DEBUG [Router]: AI Identified Questions: {q_nums if q_nums else 'None'}")
                except Exception as e:
                    print(f"DEBUG [Router]: AI Identification failed: {e}")
                    q_nums = []
            
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
    Grades identified questions by grouping those that share the same images to optimize cost.
    """
    print(f"--- [NODE: Batch Grade] Optimizing via Question Grouping ---")
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

    # 3. Group by Image Set (URL combinations)
    image_groups = {} # { tuple(sorted_urls): [q_nums] }
    for q_num, urls in state["routed_work"].items():
        if q_num not in questions_map: continue
        url_key = tuple(sorted(urls))
        if url_key not in image_groups:
            image_groups[url_key] = []
        image_groups[url_key].append(q_num)

    for urls, q_nums in image_groups.items():
        print(f"\n--- [BUNDLE] Grading Questions {q_nums} across {len(urls)} images ---")
        
        # Prepare rubrics bundle
        rubrics_text = ""
        for q_num in q_nums:
            question = questions_map[q_num]
            rubric = question.get('marking_scheme') or question.get('rubric') or "Missing Rubric."
            
            # Sum up max marks for this question
            import re
            all_marks = re.findall(r"(?:Marks:|TOTAL MARKS:)\s*(\d+)", rubric)
            max_p = sum(int(m) for m in all_marks) if all_marks else (question.get('points') or 10)
            
            rubrics_text += f"\n[RUBRIC FOR Q{q_num}] (Max Marks: {max_p}):\n{rubric}\n"

        eval_prompt = (
            f"You are a Senior WAEC Examiner. Grade the following questions based on the attached images and official rubrics.\n\n"
            f"{rubrics_text}\n"
            "INSTRUCTIONS:\n"
            "1. TRANSCRIBE the work for each question accurately.\n"
            "2. Award marks (M1, A1, B1) based on the specific rubric for each question.\n"
            "3. OUTPUT: Return a list of objects, one for each question number requested."
        )
        
        messages = [
            SystemMessage(content=(
                "Output JSON only: { 'results': [ "
                "{ 'question_number': int, 'score': int, 'summative_reasoning': 'string', 'ocr_transcript': 'string' } "
                "] }"
            )),
            HumanMessage(content=[{"type": "text", "text": eval_prompt}])
        ]
        
        for url in urls:
            messages[1].content.append({"type": "image_url", "image_url": {"url": url}})
            
        for attempt in range(3): # Retry
            try:
                import asyncio
                await asyncio.sleep(2.0)
                
                response = await llm.ainvoke(messages)
                content = response.content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                bundle_results = data.get("results", [])
                
                for res in bundle_results:
                    qn = str(res.get("question_number"))
                    score = res.get("score", 0)
                    reasoning = res.get("summative_reasoning", "No reasoning.")
                    ocr = res.get("ocr_transcript", "No OCR.")
                    
                    print(f" -> Q{qn}: {score} Marks | {reasoning[:60]}...")
                    
                    total_score += score
                    results.append({
                        "question_number": qn,
                        "score": score,
                        "summative_reasoning": reasoning,
                        "ocr": ocr
                    })
                break
            except Exception as e:
                print(f"!!! Error in Bundle {q_nums}: {e}")
                if attempt == 2: break

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
    
    print(f"Launching Batch AI Grader for attempt: {attempt_id}")
    final_output = await ace_wassce_grader.ainvoke(initial_state)
    return final_output
