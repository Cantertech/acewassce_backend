# AceWassce AI Backend

Scalable, AI-powered backend for WASSCE exam simulation and grading. This engine handles the automated marking of both MCQ and handwritten Theory papers using LangGraph and Google Gemini AI.

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup:**
   Create a `.env` file based on `.env.example`:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `GOOGLE_API_KEY` (For the AI Grader)

3. **Run the API:**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🧠 Core Features
- **MCQ Grader:** Instant server-side grading via SQL RPC.
- **Theory AI Agent:** Multi-node LangGraph workflow that analyzes handwritten images.
- **Handwriting OCR:** Uses Gemini Pro Vision to extract workings from student uploads.
- **Marking Scheme Alignment:** Compares student steps against official WAEC solution keys.

## 🛠 Tech Stack
- **Framework:** FastAPI
- **AI Orchestration:** LangGraph / LangChain
- **Database:** Supabase (PostgreSQL)
- **OCR/Vision:** Google Gemini Pro Vision
