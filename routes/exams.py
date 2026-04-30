from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query, Form
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
    is_general: str = Query("false"), 
    tags: Optional[str] = Form(None), # Capture manual tags
    file: UploadFile = File(...), 
    db=Depends(get_db)
):
    """
    Uploads an image file to Supabase Storage and saves the URL to theory_submissions.
    Uses 'feedback' column to store manual question tags if provided.
    """
    try:
        print(f"DEBUG: Starting upload for attempt {attempt_id}, general={is_general}, tags={tags}")
        
        # 1. Generate unique filename
        file_extension = file.filename.split(".")[-1]
        is_gen_bool = is_general.lower() == "true"
        folder = "general" if is_gen_bool else str(question_number)
        file_name = f"{attempt_id}/{folder}/{uuid.uuid4()}.{file_extension}"
        
        # 2. OPTIMIZATION: Compress and Resize using Pillow
        raw_content = await file.read()
        print(f"DEBUG: Received file {file.filename}, size {len(raw_content)} bytes")
        
        try:
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
            print(f"DEBUG: Compression successful. New size {len(optimized_content)} bytes")
        except Exception as img_err:
            print(f"ERROR during image processing: {str(img_err)}")
            # Fallback to original content if Pillow fails
            optimized_content = raw_content

        # 3. Upload to Supabase Storage (bucket: wassce_workings) with RETRY
        print(f"DEBUG: Uploading to Supabase bucket 'wassce_workings' at path {file_name}")
        
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                storage_response = db.storage.from_("wassce_workings").upload(
                    path=file_name,
                    file=optimized_content,
                    file_options={"content-type": "image/jpeg"}
                )
                print(f"DEBUG: Upload successful on attempt {attempt + 1}")
                break
            except Exception as upload_err:
                last_error = upload_err
                print(f"WARNING: Upload attempt {attempt + 1} failed: {str(upload_err)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1) # Wait a bit before retrying
                else:
                    raise upload_err
        
        # 3. Get Public URL
        image_url = db.storage.from_("wassce_workings").get_public_url(file_name)
        
        # 4. Save to database
        theory_data = {
            "attempt_id": attempt_id,
            "question_number": question_number,
            "image_url": image_url,
            "is_general": is_gen_bool,
            "feedback": tags # Store manual tags here for the router to use
        }
        db.table("theory_submissions").insert(theory_data).execute()
        
        print(f"DEBUG: Upload and database record created successfully for {file_name}")
        return {"status": "success", "image_url": image_url}
    except Exception as e:
        print(f"CRITICAL ERROR in upload_working: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{attempt_id}/grade")
async def trigger_grading(attempt_id: str, background_tasks: BackgroundTasks, db=Depends(get_db)):
    """
    Triggers the LangGraph AI grading workflow asynchronously for all theory submissions in an attempt.
    """
    try:
        # 1. Fetch all submissions for this attempt
        response = db.table("theory_submissions").select("id", "image_url", "question_id", "question_number", "is_general", "feedback").eq("attempt_id", attempt_id).execute()
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
    Background worker to grade each theory submission and aggregate scores.
    Dynamic Logic: Uses exam-specific compulsory_questions and marking_config.
    """
    try:
        # 1. Fetch current attempt and its exam configuration
        attempt_res = db.table("exam_attempts").select("total_score, status, exam_id").eq("id", attempt_id).single().execute()
        current_total = attempt_res.data.get("total_score", 0)
        exam_id = attempt_res.data.get("exam_id")
        
        exam_res = db.table("exams").select("*").eq("id", exam_id).single().execute()
        exam_data = exam_res.data or {}
        
        # Marking Configuration (Defaults to WAEC Standard)
        compulsory_count = exam_data.get('compulsory_questions', 5)
        elective_pick = 5  # Standard for most WAEC Part B
        section_a_max = 40
        section_b_max = 60
        mcq_max = 50
        total_possible = section_a_max + section_b_max + mcq_max # 150
        result = await run_grader(
            attempt_id=attempt_id,
            submissions=submissions
        )
        
        grading_results = result.get("grading_results", [])
        
        # Section A/B Logic
        part_a_score = 0
        part_b_scores = []
        
        for res in grading_results:
            try:
                q_num = int(res.get("question_number", 0))
                score = res.get("score", 0)
                
                if 1 <= q_num <= compulsory_count:
                    part_a_score += score
                else:
                    part_b_scores.append(score)
            except (ValueError, TypeError):
                continue
        
        # Take top X from Section B (Elective Pick)
        part_b_scores.sort(reverse=True)
        part_b_score = sum(part_b_scores[:elective_pick])
        
        ai_theory_score = part_a_score + part_b_score
        
        # 3. Aggregate with MCQ
        resp_res = db.table("exam_responses").select("id", count="exact").eq("attempt_id", attempt_id).execute()
        has_mcq = resp_res.count > 0 if resp_res.count is not None else False
        
        mcq_score = current_total 
        raw_grand_total = mcq_score + ai_theory_score
        
        # Scale to 100 based on total_possible
        final_percentage = round((raw_grand_total / total_possible) * 100)
        
        # 4. Update status
        new_status = "graded" if has_mcq else "theory_marked"
        
        db.table("exam_attempts").update({
            "status": new_status,
            "total_score": final_percentage,
            "end_time": datetime.utcnow().isoformat()
        }).eq("id", attempt_id).execute()
        
        print(f"--- GRADING COMPLETE: Q1-5: {part_a_score}, Q6-13 (Top 5): {part_b_score}, Theory: {ai_theory_score}, MCQ: {mcq_score}, Final: {final_percentage}%, Status: {new_status} ---")
        
    except Exception as e:
        print(f"CRITICAL ERROR in process_full_attempt_grading: {str(e)}")
        import traceback
        traceback.print_exc()
