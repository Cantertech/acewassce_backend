from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid
from backend.database import get_db
from backend.ai_engine.agent import run_grader

router = APIRouter(prefix="/api/v1/attempts", tags=["Exam Attempts"])

class AttemptStartRequest(BaseModel):
    student_id: str
    exam_id: str

class AttemptResponse(BaseModel):
    id: str
    student_id: str
    exam_id: str
    start_time: datetime
    status: str

@router.post("/start", response_model=AttemptResponse)
async def start_attempt(request: AttemptStartRequest, db=Depends(get_db)):
    """
    Creates a new row in exam_attempts.
    """
    try:
        data = {
            "student_id": request.student_id,
            "exam_id": request.exam_id,
            "status": "in_progress",
            "start_time": datetime.utcnow().isoformat()
        }
        response = db.table("exam_attempts").insert(data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to create attempt")
            
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{attempt_id}/upload-working")
async def upload_working(attempt_id: str, question_number: int, file: UploadFile = File(...), db=Depends(get_db)):
    """
    Uploads an image file to Supabase Storage and saves the URL to theory_submissions.
    """
    try:
        # 1. Generate unique filename
        file_extension = file.filename.split(".")[-1]
        file_name = f"{attempt_id}/{question_number}_{uuid.uuid4()}.{file_extension}"
        
        # 2. Upload to Supabase Storage (bucket: wassce_workings)
        content = await file.read()
        storage_response = db.storage.from_("wassce_workings").upload(
            path=file_name,
            file=content,
            file_options={"content-type": file.content_type}
        )
        
        # 3. Get Public URL
        image_url = db.storage.from_("wassce_workings").get_public_url(file_name)
        
        # 4. Save to database
        theory_data = {
            "attempt_id": attempt_id,
            "question_number": question_number,
            "image_url": image_url
        }
        db.table("theory_submissions").insert(theory_data).execute()
        
        return {"status": "success", "image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{attempt_id}/grade")
async def trigger_grading(attempt_id: str, background_tasks: BackgroundTasks, db=Depends(get_db)):
    """
    Triggers the LangGraph AI grading workflow asynchronously for all theory submissions in an attempt.
    """
    try:
        # 1. Fetch all submissions for this attempt
        response = db.table("theory_submissions").select("id", "image_url", "question_id").eq("attempt_id", attempt_id).execute()
        submissions = response.data
        
        if not submissions:
            raise HTTPException(status_code=404, detail="No theory submissions found for this attempt")

        # 2. Add grading task to background
        background_tasks.add_task(process_full_attempt_grading, attempt_id, submissions, db)
        
        return {"status": "grading_started", "message": f"AI Grading workflow launched for attempt {attempt_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_full_attempt_grading(attempt_id: str, submissions: List[dict], db):
    """
    Background worker to grade each theory submission using the LangGraph agent.
    """
    total_score = 0
    try:
        for sub in submissions:
            print(f"Propagating grading for submission: {sub['id']}")
            # Each call to run_grader represents one LangGraph workflow execution
            # We pass the submission record ID as the attempt_id for the graph's saving node
            result = await run_grader(
                attempt_id=sub["id"], 
                question_id=sub["question_id"],
                image_url=sub["image_url"]
            )
            total_score += result.get("final_score", 0)
        
        # Finally, update the main exam_attempt status to 'graded'
        db.table("exam_attempts").update({
            "status": "graded",
            "total_score": total_score,
            "end_time": datetime.utcnow().isoformat()
        }).eq("id", attempt_id).execute()
        
        print(f"--- ALL GRADED: Attempt {attempt_id} completed with total score: {total_score} ---")
    except Exception as e:
        print(f"CRITICAL ERROR in process_full_attempt_grading: {str(e)}")
