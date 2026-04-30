import os
import json

SCHEMES_DIR = r"c:\Users\silas\Desktop\acewassce-mock-master\data_pipeline\processed\schemes"

def forensic_map():
    print("--- [FORENSIC MAP] JSON Filename vs Internal Question ID ---")
    files = [f for f in os.listdir(SCHEMES_DIR) if f.endswith(".json") and "CORE 2" in f]
    
    for filename in files:
        filepath = os.path.join(SCHEMES_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                schemes = data.get("schemes", [])
                q_ids = [str(s.get("question_id")) for s in schemes]
                print(f"{filename.ljust(50)} -> {q_ids}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    forensic_map()
