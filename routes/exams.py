from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid
import io
from PIL import Image
from database import get_db
from ai_engine.agent import run_grader

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
async def upload_working(
    attempt_id: str, 
    question_number: Optional[int] = Query(None), 
    is_general: bool = Query(False), 
    file: UploadFile = File(...), 
    db=Depends(get_db)
):
    """
    Uploads an image file to Supabase Storage and saves the URL to theory_submissions.
    """
    try:
        # 1. Generate unique filename
        file_extension = file.filename.split(".")[-1]
        folder = "general" if is_general else str(question_number)
        file_name = f"{attempt_id}/{folder}/{uuid.uuid4()}.{file_extension}"
        
        # 2. OPTIMIZATION: Compress and Resize using Pillow
        raw_content = await file.read()
        img = Image.open(io.BytesIO(raw_content))
        
        # Convert to RGB if necessary (to handle PNG/RGBA)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Resize if too large (Max width 1200px)
        max_size = 1200
        if img.width > max_size:
            ratio = max_size / float(img.width)
            new_height = int(float(img.height) * float(ratio))
            img = img.resize((max_size, new_height), Image.Resampling.LANCZOS)
            
        # Save to buffer with high compression (Quality 70)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70, optimize=True)
        optimized_content = buffer.getvalue()
        
        # 3. Upload to Supabase Storage (bucket: wassce_workings)
        storage_response = db.storage.from_("wassce_workings").upload(
            path=file_name,
            file=optimized_content,
            file_options={"content-type": "image/jpeg"}
        )
        
        # 3. Get Public URL
        image_url = db.storage.from_("wassce_workings").get_public_url(file_name)
        
        # 4. Save to database
        theory_data = {
            "attempt_id": attempt_id,
            "question_number": question_number,
            "image_url": image_url,
            "is_general": is_general
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
        response = db.table("theory_submissions").select("id", "image_url", "question_id", "question_number", "is_general").eq("attempt_id", attempt_id).execute()
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
        # In the new flow, we pass ALL submissions to one run_grader call 
        # which will perform the routing and batch grading.
        result = await run_grader(
            attempt_id=attempt_id,
            submissions=submissions
        )
        total_score = result.get("total_score", 0)
        
        # Finally, update the main exam_attempt status to 'graded'
        db.table("exam_attempts").update({
            "status": "graded",
            "total_score": total_score,
            "end_time": datetime.utcnow().isoformat()
        }).eq("id", attempt_id).execute()
        
        print(f"--- ALL GRADED: Attempt {attempt_id} completed with total score: {total_score} ---")
    except Exception as e:
        print(f"CRITICAL ERROR in process_full_attempt_grading: {str(e)}")
