import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import exams

app = FastAPI(
    title="AceWassce Backend API",
    description="Scalable, AI-powered backend for exam simulation and grading.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(exams.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "AceWassce Backend API is running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
