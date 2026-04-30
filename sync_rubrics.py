import os
import json
import asyncio
import re
from database import get_db

EXAM_ID = "fd9c561c-f0f1-411d-aee7-75f2656f4904"
SCHEMES_DIR = r"c:\Users\silas\Desktop\acewassce-mock-master\data_pipeline\processed\schemes"

async def sync_theory_rubrics():
    print(f"--- [SYNC START] Injecting OFFICIAL THEORY RUBRICS (Core 2) ---")
    db = get_db()
    
    # ONLY pick Core 2 files for Theory questions
    files = [f for f in os.listdir(SCHEMES_DIR) if f.endswith(".json") and "CORE 2" in f]
    print(f"DEBUG: Found {len(files)} Core 2 JSON files.")
    
    rubric_groups = {}
    
    for filename in files:
        filepath = os.path.join(SCHEMES_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        schemes = data.get("schemes", [])
        for scheme in schemes:
            q_id_raw = str(scheme.get("question_id"))
            match = re.match(r"(\d+)", q_id_raw)
            if not match: continue
            q_num_main = match.group(1)
            
            part_lines = []
            part_lines.append(f"--- SUB-PART {q_id_raw} (Marks: {scheme.get('total_marks')}) ---")
            for step in scheme.get("steps", []):
                line = f"- Step {step['step_order']}: {step['expected_logic']} | Equation: {step['expected_equation']} | Mark: {step['mark_type']}({step['marks_awarded']})"
                part_lines.append(line)
            
            sub_rubric = "\n".join(part_lines)
            
            if q_num_main not in rubric_groups:
                rubric_groups[q_num_main] = []
            rubric_groups[q_num_main].append(sub_rubric)

    sync_count = 0
    for q_num, parts in rubric_groups.items():
        # Sort parts so 1(a) comes before 1(b)
        parts.sort()
        full_rubric = "\n\n".join(parts)
        
        try:
            res = db.table("questions").select("id").eq("exam_id", EXAM_ID).eq("question_number", q_num).eq("is_mcq", False).execute()
            
            if res.data:
                q_db_id = res.data[0]["id"]
                db.table("questions").update({"marking_scheme": full_rubric}).eq("id", q_db_id).execute()
                print(f"SUCCESS: Question {q_num} (Theory) updated with {len(parts)} detailed parts.")
                sync_count += 1
            else:
                print(f"SKIP: Theory Question {q_num} not found in database.")
        except Exception as e:
            print(f"Error syncing Question {q_num}: {e}")

    print(f"\n--- [THEORY SYNC COMPLETE] Total Questions Updated: {sync_count} ---")

if __name__ == "__main__":
    asyncio.run(sync_theory_rubrics())
