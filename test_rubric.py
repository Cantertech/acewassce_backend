import os
import asyncio
from database import get_db

async def test_rubric_retrieval():
    print("--- [DATABASE PROBE] Testing Rubric Retrieval ---")
    db = get_db()
    
    try:
        response = db.table("questions").select("*").eq("is_mcq", False).limit(3).execute()
        questions = response.data
        
        if not questions:
            print("RESULT: No theory questions found in the database.")
            return
            
        for q in questions:
            q_num = q.get("question_number")
            rubric = q.get("marking_scheme") or q.get("rubric") or q.get("marking_guide")
            
            print(f"\n[Question {q_num}]")
            if rubric:
                print(f"FOUND: Rubric detected (First 150 chars): {rubric[:150]}...")
            else:
                print("MISSING: Rubric columns are all null.")
                
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_rubric_retrieval())
