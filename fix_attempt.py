import asyncio
from database import get_db
from routes.exams import aggregate_and_finalize_scores, process_full_attempt_grading

async def fix_attempt(attempt_id):
    db = get_db()
    print(f"Fixing attempt: {attempt_id}")
    
    # 1. Fetch submissions
    res = db.table("theory_submissions").select("*").eq("attempt_id", attempt_id).execute()
    submissions = res.data
    
    if not submissions:
        print("No submissions found for this attempt.")
        return

    # 2. Re-trigger full grading workflow
    print("Re-launching grading pipeline...")
    await process_full_attempt_grading(attempt_id, submissions, db)
    print("Done! Check the results page now.")

if __name__ == "__main__":
    import sys
    target_id = "0087a51e-5577-438c-9eaf-2904f86af6e0"
    asyncio.run(fix_attempt(target_id))
