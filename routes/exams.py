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

async def aggregate_and_finalize_scores(attempt_id: str, db):
    """
    Combines raw Theory and MCQ scores, scales them to 100, and updates the DB.
    """
    try:
        # 1. Fetch granular scores
        attempt_res = db.table("exam_attempts").select("mcq_score, theory_score, exam_id").eq("id", attempt_id).single().execute()
        mcq_raw = attempt_res.data.get("mcq_score", 0)
        theory_raw = attempt_res.data.get("theory_score", 0)
        exam_id = attempt_res.data.get("exam_id")
        
        exam_res = db.table("exams").select("*").eq("id", exam_id).single().execute()
        exam_data = exam_res.data or {}
        
        section_a_max = 40
        section_b_max = 60
        mcq_max = 50
        total_possible = section_a_max + section_b_max + mcq_max # 150
        
        raw_grand_total = mcq_raw + theory_raw
        final_percentage = round((raw_grand_total / total_possible) * 100)
        
        # 2. Update status and final aggregate score
        db.table("exam_attempts").update({
            "status": "graded",
            "total_score": final_percentage,
            "end_time": datetime.utcnow().isoformat()
        }).eq("id", attempt_id).execute()
        
        print(f"--- AGGREGATION COMPLETE: Theory: {theory_raw}, MCQ: {mcq_raw}, Final: {final_percentage}% ---")
        return final_percentage
    except Exception as e:
        print(f"Aggregation Error: {e}")
        return 0

async def process_full_attempt_grading(attempt_id: str, submissions: List[dict], db):
    """
    Background worker to grade theory and store in the new granular columns.
    """
    try:
        # 1. Fetch exam config
        attempt_res = db.table("exam_attempts").select("exam_id, status").eq("id", attempt_id).single().execute()
        exam_id = attempt_res.data.get("exam_id")
        current_status = attempt_res.data.get("status")
        
        exam_res = db.table("exams").select("compulsory_questions").eq("id", exam_id).single().execute()
        compulsory_count = exam_res.data.get('compulsory_questions', 5)
        
        # 2. Run AI Grader
        result = await run_grader(attempt_id=attempt_id, submissions=submissions)
        grading_results = result.get("grading_results", [])
        
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
            except: continue
        
        part_b_scores.sort(reverse=True)
        part_b_score = sum(part_b_scores[:5])
        ai_theory_score = part_a_score + part_b_score
        
        # 3. Update Granular Theory Score
        db.table("exam_attempts").update({
            "theory_score": ai_theory_score,
            "total_theory": 100,
            "status": "theory_marked" if current_status != "mcq_marked" else "graded"
        }).eq("id", attempt_id).execute()
        
        # 4. Finalize if both are done
        # Check if MCQ is marked or has responses
        resp_res = db.table("exam_responses").select("id", count="exact").eq("attempt_id", attempt_id).execute()
        if (resp_res.count or 0) > 0 or current_status == "mcq_marked":
            await aggregate_and_finalize_scores(attempt_id, db)
        else:
            print(f"Theory Grading Done: {ai_theory_score} points. Waiting for MCQs.")

    except Exception as e:
        print(f"CRITICAL ERROR in process_full_attempt_grading: {str(e)}")

@router.post("/{attempt_id}/grade-mcq")
async def grade_mcq(attempt_id: str, db=Depends(get_db)):
    """
    Grades MCQ and stores results in the new granular columns.
    """
    print(f"🚀 GRADING MCQs for attempt: {attempt_id}")
    try:
        # 1. Fetch student responses
        res = db.table("exam_responses").select("*").eq("attempt_id", attempt_id).execute()
        responses = res.data
        if not responses:
            return {"status": "success", "mcq_score": 0, "total_mcq": 0}

        attempt_res = db.table("exam_attempts").select("exam_id, status").eq("id", attempt_id).single().execute()
        exam_id = attempt_res.data["exam_id"]
        current_status = attempt_res.data["status"]
        
        q_res = db.table("questions").select("id, marking_scheme").eq("exam_id", exam_id).eq("is_mcq", True).execute()
        questions_map = {q["id"]: q["marking_scheme"] for q in q_res.data}

        score = 0
        import re
        for resp in responses:
            q_id = resp["question_id"]
            student_choice = resp["selected_option"]
            marking = questions_map.get(q_id, "")
            match = re.search(r"Equation:\s*([A-D])\s*=", marking)
            if match and student_choice == match.group(1):
                score += 1
            elif f" {student_choice} =" in marking:
                score += 1

        # 2. Update Granular MCQ Score
        db.table("exam_attempts").update({
            "mcq_score": score,
            "total_mcq": len(q_res.data),
            "status": "mcq_marked" if current_status != "theory_marked" else "graded"
        }).eq("id", attempt_id).execute()

        # 3. Finalize if Theory is done
        theory_res = db.table("theory_submissions").select("id", count="exact").eq("attempt_id", attempt_id).execute()
        if (theory_res.count or 0) > 0 or current_status == "theory_marked":
            await aggregate_and_finalize_scores(attempt_id, db)
            return {"status": "success", "mcq_score": score, "total_mcq": len(q_res.data), "finalized": True}
        
        return {"status": "success", "mcq_score": score, "total_mcq": len(q_res.data), "finalized": False}

    except Exception as e:
        print(f"ERROR grading MCQs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
