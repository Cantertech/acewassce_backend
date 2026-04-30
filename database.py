import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class SupabaseManager:
    _client: Client = None

    @classmethod
    def get_client(cls) -> Client:
        if cls._client is None:
            if not SUPABASE_URL or not SUPABASE_KEY:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            
            # Safe Debug: Show last 4 chars of the key being used
            print(f"--- [AUTH DEBUG] Using Supabase Key ending in: ...{SUPABASE_KEY[-4:]} ---")
            
            cls._client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return cls._client

def get_db():
    return SupabaseManager.get_client()
