import asyncio
import sys
import os

# Add the current directory to sys.path so we can import 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai_engine.agent import run_grading_agent

async def test_grading_workflow():
    print("🚀 Initializing AI Workflow Test...")
    
    # Mock data
    sample_attempt_id = "test-attempt-123"
    sample_images = [
        "https://example.com/workings/math_q1.jpg",
        "https://example.com/workings/math_q2.jpg"
    ]
    
    print(f"📝 Testing with Attempt ID: {sample_attempt_id}")
    print(f"🖼️ Sample Images: {len(sample_images)}")
    print("-" * 30)
    
    # Run the agent
    try:
        final_state = await run_grading_agent(sample_attempt_id, sample_images)
        
        print("-" * 30)
        print("✅ Workflow Completed Successfully!")
        print(f"📊 Final Score: {final_state.get('final_score')}")
        print(f"💬 Feedback: {final_state.get('feedback')}")
        print(f"📄 Extracted Text: {final_state.get('extracted_text')[:50]}...") # Show first 50 chars
        
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")

if __name__ == "__main__":
    asyncio.run(test_grading_workflow())
