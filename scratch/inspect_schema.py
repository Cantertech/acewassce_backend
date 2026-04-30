import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Get one attempt
res = supabase.table("exam_attempts").select("*").limit(1).execute()
if res.data:
    print("Columns in 'exam_attempts':", res.data[0].keys())
else:
    print("No attempts found.")
