import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

# Import the agent components
# Note: we are importing specific nodes to test the logic without needing a live Supabase connection yet
from ai_engine.agent import evaluate_steps_node, generate_feedback_node, GradingState

# Load environment variables from potential locations
possible_envs = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent / "data_pipeline" / ".env",
    Path(__file__).parent / ".env"
]
for env_path in possible_envs:
    if env_path.exists():
        print(f"📡 Loading config from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        break
else:
    print("⚠️ No .env file found in expected locations.")

async def test_end_to_end_grading():
    print("🚀 Starting End-to-End Agent Test with 2025 Rubric...")
    
    # 1. Load the official 2025 Rubric we just extracted
    data_dir = Path(__file__).parent.parent / "data_pipeline"
    scheme_file = data_dir / "sample_scheme_output.json"
    
    with open(scheme_file, "r") as f:
        schemes_data = json.load(f)
        # We'll use the scheme for Question 1 (a)
        official_rubric = json.dumps(schemes_data["schemes"][0], indent=2)

    # 2. Simulate a Student's handwritten answer (transcribed)
    # The student gets Part A right but makes a mistake in Part B
    student_workings = """
    [STUDENT WORKING]
    Question 1:
    (a) Average speed.
    Total distance = 300km.
    First half distance = 150km. Speed = 60km/h. Time t1 = 150/60 = 2.5 hours.
    Second half distance = 150km. Speed = 90km/h. Time t2 = 150/90 = 1.67 hours.
    Total time = 2.5 + 1.67 = 4.17 hours.
    Average speed = 300 / 4.17 = 71.94 km/h. Rounding to 72 km/h.
    
    (b) If speed is 50km/h.
    New t1 = 150 / 50 = 3 hours.
    Wait, I think the time won't change much. Maybe 5 minutes more?
    [END]
    """

    # 3. Create the initial state
    state: GradingState = {
        "attempt_id": "test_attempt_123",
        "question_id": "2025_math_1",
        "image_url": "mock_url",
        "rubric": official_rubric,
        "extracted_steps": student_workings,
        "evaluation_matrix": {},
        "final_score": 0,
        "tutor_feedback": ""
    }

    print("\n--- STEP 1: EVALUATING STUDENT STEPS ---")
    state = await evaluate_steps_node(state)
    print(f"Final Score Awarded: {state['final_score']}")
    print("Evaluation Breakdown:")
    print(json.dumps(state['evaluation_matrix'], indent=2))

    print("\n--- STEP 2: GENERATING TUTOR FEEDBACK ---")
    state = await generate_feedback_node(state)
    print("AI Tutor Response:")
    print(f"\"{state['tutor_feedback']}\"")

    print("\n✅ Test Complete!")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
    else:
        asyncio.run(test_end_to_end_grading())
