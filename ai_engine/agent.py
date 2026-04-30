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
    Represents the state of the grading workflow.
    """
    attempt_id: str
    question_id: str
    image_url: str
    rubric: str
    extracted_steps: str
    evaluation_matrix: Dict
    final_score: int
    tutor_feedback: str

# --- 2. THE NODES ---

async def extract_math_node(state: GradingState):
    """
    OCR & Image Analysis Node: Transcribes math and describes diagrams factually using Gemini 1.5 Flash.
    """
    print(f"--- [NODE: Extract Math] Starting OCR & Diagram Analysis for image: {state['image_url']} ---")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = (
            "You are an expert OCR and image analysis system. Read this handwritten math and transcribe it "
            "line-by-line exactly as written. If the student has drawn a diagram, sketch, or shape, write a "
            "strict, factual description of it under a [DIAGRAM] tag. State exactly what shapes are drawn, "
            "what labels or measurements exist, and where they are located relative to each other "
            "(e.g., 'A right-angled triangle with 100m on the vertical axis and 48 degrees at the top vertex'). "
            "Do not solve the math or grade it. Just transcribe the text and describe the imagery."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": state["image_url"]},
            ]
        )
        
        response = await llm.ainvoke([message])
        state["extracted_steps"] = response.content
        return state
    except Exception as e:
        print(f"Error in extract_math_node: {e}")
        state["extracted_steps"] = "OCR Failed"
        return state

async def fetch_rubric_node(state: GradingState):
    """
    Database Node: Fetches the marking scheme from Supabase.
    """
    print(f"--- [NODE: Fetch Rubric] Fetching rubric for question: {state['question_id']} ---")
    try:
        db = get_db()
        response = db.table("theory_questions").select("marking_scheme").eq("id", state["question_id"]).execute()
        
        if response.data:
            state["rubric"] = response.data[0]["marking_scheme"]
        else:
            state["rubric"] = "No rubric found in database."
        return state
    except Exception as e:
        print(f"Error in fetch_rubric_node: {e}")
        state["rubric"] = "Error fetching rubric."
        return state

async def evaluate_steps_node(state: GradingState):
    """
    Grading Node: Compares extracted steps (including [DIAGRAM] tags) against the rubric.
    """
    print(f"--- [NODE: Evaluate Steps] Comparing transcription & diagrams against rubric ---")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = (
            "You are a strict high school math examiner. You will receive the student's extracted working "
            "(which may include a [DIAGRAM] text description of their drawing) and the official WAEC JSON marking scheme. "
            "Compare the student's steps and diagram descriptions against the rubric. Award Method Marks (M1), "
            "Accuracy Marks (A1), and Independent Marks (B1 for diagrams). Output your evaluation strictly as "
            "a JSON object containing a step-by-step breakdown of the marks and the total final_score."
        )
        
        prompt = (
            f"RUBRIC:\n{state['rubric']}\n\n"
            f"STUDENT WORKINGS (with [DIAGRAM] tags):\n{state['extracted_steps']}\n\n"
            "Evaluate and return JSON."
        )
        
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        
        # Parse JSON
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        state["evaluation_matrix"] = result.get("breakdown", {})
        state["final_score"] = result.get("total_score", 0)
        return state
    except Exception as e:
        print(f"Error in evaluate_steps_node: {e}")
        state["evaluation_matrix"] = {"error": "Grading failed"}
        state["final_score"] = 0
        return state

async def generate_feedback_node(state: GradingState):
    """
    Tutor Node: Generates empathetic feedback based on the evaluation result.
    """
    print(f"--- [NODE: Generate Feedback] Crafting supportive message ---")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = (
            "You are a highly supportive and empathetic math tutor. Look at the student's evaluation matrix. "
            "Write a brief, highly encouraging message explaining exactly where they went wrong, praising what they did right, "
            "and offering a quick tip to fix their mistake. Speak directly to the student."
        )
        
        prompt = (
            f"EVALUATION MATRIX: {json.dumps(state['evaluation_matrix'])}\n"
            f"FINAL SCORE: {state['final_score']}\n"
            f"STUDENT WORKINGS: {state['extracted_steps']}\n"
        )
        
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        state["tutor_feedback"] = response.content
        return state
    except Exception as e:
        print(f"Error in generate_feedback_node: {e}")
        state["tutor_feedback"] = "Keep trying! You are doing great."
        return state

async def save_results_node(state: GradingState):
    """
    Database Node: Saves the final score and feedback to the theory_submissions table.
    """
    print(f"--- [NODE: Save Results] Persisting results for attempt: {state['attempt_id']} ---")
    try:
        db = get_db()
        data = {
            "ai_score": state["final_score"],
            "ai_feedback": state["tutor_feedback"],
            # Assuming evaluation_matrix can be stored in a JSONB column or similar
            # metadata: state["evaluation_matrix"] 
        }
        
        # Note: In our current schema, theory_submissions uses 'id' (UUID) which corresponds to the submission entry
        # We need to make sure 'attempt_id' here refers to the record ID in theory_submissions
        db.table("theory_submissions").update(data).eq("id", state["attempt_id"]).execute()
        return state
    except Exception as e:
        print(f"Error in save_results_node: {e}")
        return state

# --- 3. THE GRAPH COMPILATION ---

workflow = StateGraph(GradingState)

# Add Nodes
workflow.add_node("extract_math", extract_math_node)
workflow.add_node("fetch_rubric", fetch_rubric_node)
workflow.add_node("evaluate_steps", evaluate_steps_node)
workflow.add_node("generate_feedback", generate_feedback_node)
workflow.add_node("save_results", save_results_node)

# Build Edges
workflow.add_edge(START, "extract_math")
workflow.add_edge("extract_math", "fetch_rubric")
workflow.add_edge("fetch_rubric", "evaluate_steps")
workflow.add_edge("evaluate_steps", "generate_feedback")
workflow.add_edge("generate_feedback", "save_results")
workflow.add_edge("save_results", END)

# Compile
ace_wassce_grader = workflow.compile()

async def run_grader(attempt_id: str, question_id: str, image_url: str):
    """
    Wrapper function to invoke the grading graph.
    """
    initial_state = {
        "attempt_id": attempt_id,
        "question_id": question_id,
        "image_url": image_url,
        "rubric": "",
        "extracted_steps": "",
        "evaluation_matrix": {},
        "final_score": 0,
        "tutor_feedback": ""
    }
    
    print(f"🚀 Launching AceWassce Grader for attempt: {attempt_id}")
    final_output = await ace_wassce_grader.ainvoke(initial_state)
    return final_output
