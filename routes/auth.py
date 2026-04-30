import os
import httpx
import random
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from backend.database import get_db

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# Hubtel Configuration from .env
HUBTEL_CLIENT_ID = os.getenv("HUBTEL_CLIENT_ID")
HUBTEL_CLIENT_SECRET = os.getenv("HUBTEL_CLIENT_SECRET")
HUBTEL_SENDER_ID = os.getenv("HUBTEL_SENDER_ID", "AceWassce")

class OTPRequest(BaseModel):
    phone_number: str # Format: 24xxxxxxx

class OTPVerify(BaseModel):
    phone_number: str
    otp_code: str

@router.post("/send-otp")
async def send_otp(request: OTPRequest, db=Depends(get_db)):
    """
    Generates a 4-digit OTP and sends it via Hubtel SMS.
    """
    otp = str(random.randint(1000, 9999))
    message = f"Your AceWassce login code is: {otp}. Valid for 5 minutes."
    
    # 1. Store OTP in Database (temporary table or user meta)
    # For now, let's assume we update a 'students' table with the otp
    try:
        db.table("students").upsert({
            "phone_number": request.phone_number,
            "current_otp": otp
        }, on_conflict="phone_number").execute()
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate OTP")

    # 2. Send SMS via Hubtel API
    url = f"https://smsc.hubtel.com/v1/messages/send"
    params = {
        "clientid": HUBTEL_CLIENT_ID,
        "clientsecret": HUBTEL_CLIENT_SECRET,
        "from": HUBTEL_SENDER_ID,
        "to": request.phone_number,
        "content": message
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code == 201 or response.status_code == 200:
            return {"status": "success", "message": "OTP sent successfully"}
        else:
            print(f"Hubtel Error: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to send SMS")

@router.post("/verify-otp")
async def verify_otp(request: OTPVerify, db=Depends(get_db)):
    """
    Verifies the OTP and returns a session/token.
    """
    response = db.table("students").select("*").eq("phone_number", request.phone_number).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
        
    user = response.data[0]
    if user.get("current_otp") == request.otp_code:
        # Clear OTP after success
        db.table("students").update({"current_otp": None}).eq("phone_number", request.phone_number).execute()
        return {"status": "success", "token": "mock-jwt-token", "user": user}
    else:
        raise HTTPException(status_code=400, detail="Invalid OTP code")
