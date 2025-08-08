from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import requests
import os
from io import BytesIO
import fitz  # PyMuPDF
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="HealthInsure AI Backend",
    description="API to support the Health Insurance Assistant Portal.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class AskRequest(BaseModel):
    question: str
    document_context: Optional[str] = None

class AskResponse(BaseModel):
    answer: str

class UploadDocResponse(BaseModel):
    status: str
    filename: str
    summary: str
    full_text: str

class ClaimCheckResponse(BaseModel):
    decision: str
    reason: str
    required_documents: List[str]

class PolicyRecommendation(BaseModel):
    policy_name: str
    reasoning: str
    key_features: List[str]
    estimated_premium: str

class RecommendPolicyResponse(BaseModel):
    recommendations: List[PolicyRecommendation]

class Hospital(BaseModel):
    name: str
    address: str
    specialties: List[str]

class FindHospitalsResponse(BaseModel):
    hospitals: List[Hospital]

class AnalyticsResponse(BaseModel):
    claim_status_distribution: dict
    monthly_claims: dict
    key_metrics: dict

# --- Gemini API Helper ---
def call_gemini_api(prompt: str, is_json_output: bool = False):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    generation_config = {}
    if is_json_output:
        generation_config["response_mime_type"] = "application/json"

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generation_config": generation_config
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=90)
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and result["candidates"]:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in result:
            raise HTTPException(status_code=500, detail=f"Gemini API Error: {result['error'].get('message', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail="Invalid response structure from Gemini API.")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to AI service: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Routes ---
@app.post("/ask", response_model=AskResponse)
async def ask_question(data: AskRequest):
    if data.document_context:
        prompt = f"""
        You are a helpful insurance assistant. A user uploaded a policy and asked:
        Context:
        ---
        {data.document_context}
        ---
        Question: {data.question}
        Answer:
        """
    else:
        prompt = f"""
        You are a helpful insurance assistant. The user asked a general health insurance question:
        Question: {data.question}
        Answer:
        """

    answer = call_gemini_api(prompt)
    return {"answer": answer}

@app.post("/upload-doc", response_model=UploadDocResponse)
async def upload_doc(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    contents = await file.read()
    try:
        with fitz.open(stream=BytesIO(contents), filetype="pdf") as doc:
            text = "".join([page.get_text() for page in doc]).strip()
        if not text:
            raise HTTPException(status_code=400, detail="PDF is empty or non-readable.")

        summary_prompt = f"Summarize the following insurance policy in bullet points:\n\n{text[:8000]}"
        summary = call_gemini_api(summary_prompt)

        return {
            "status": "success",
            "filename": file.filename,
            "summary": summary,
            "full_text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

@app.post("/check-claim", response_model=ClaimCheckResponse)
async def check_claim(claim_type: str = Form(...), document_context: str = Form(...), bill: UploadFile = File(None)):
    prompt = f"""
    You are an expert insurance claim analyst. Evaluate the following claim based on the policy:

    Context:
    ---
    {document_context}
    ---
    Claim Type: {claim_type}
    Bill Attached: {'Yes' if bill else 'No'}

    Return JSON with:
    - decision: Eligible/Ineligible/More Information Needed
    - reason: explanation
    - required_documents: list of required documents

    JSON:
    """
    response_text = call_gemini_api(prompt, is_json_output=True)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON.")

@app.post("/recommend-policy", response_model=RecommendPolicyResponse)
async def recommend_policy(user_age: str = Form(...), user_gender: str = Form(...), health_conditions: str = Form(...), coverage: str = Form(...), budget: str = Form(...)):
    prompt = f"""
    Suggest 2-3 health insurance policies for:
    - Age: {user_age}
    - Gender: {user_gender}
    - Conditions: {health_conditions}
    - Coverage: {coverage}
    - Budget: ₹{budget}

    Return JSON with:
    recommendations: [
      {{policy_name, reasoning, key_features (list), estimated_premium}}
    ]
    """
    response_text = call_gemini_api(prompt, is_json_output=True)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON.")

@app.get("/find-hospitals", response_model=FindHospitalsResponse)
async def find_hospitals(location: str = Query(...)):
    prompt = f"""
    Find up to 5 major cashless network hospitals in "{location}" (India).
    Return JSON with:
    hospitals: [{{ name, address, specialties (list) }}]
    """
    response_text = call_gemini_api(prompt, is_json_output=True)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON.")

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics_data():
    prompt = """
    Generate realistic health insurance analytics data for an Indian dashboard:

    Return valid JSON:
    {
      "claim_status_distribution": {
        "labels": ["Approved", "Pending", "Rejected"],
        "data": [int, int, int]
      },
      "monthly_claims": {
        "labels": ["March", "April", "May", "June", "July", "August"],
        "data": [int, int, int, int, int, int]
      },
      "key_metrics": {
        "total_claims": int,
        "approval_rate": float,
        "avg_claim_amount": int
      }
    }
    """
    response_text = call_gemini_api(prompt, is_json_output=True)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON.")