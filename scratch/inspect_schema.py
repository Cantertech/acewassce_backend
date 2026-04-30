import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Get one MCQ question
res = supabase.table("questions").select("*").eq("is_mcq", True).limit(1).execute()
if res.data:
    q = res.data[0]
    print("Question Text:", q.get("question_text"))
    print("Options:", q.get("options"))
    print("Marking Scheme:", q.get("marking_scheme"))
else:
    print("No MCQ questions found.")
