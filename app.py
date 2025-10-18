from typing import List, Dict
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

# ===================== CONFIGURATION =====================

app = FastAPI(
    title="Chat API",
    version="0.1.0",
    description="An API for a friendly chatbot using Hugging Face InferenceClient. Swagger UI is available at /swagger.",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("api_server")

MODEL_ID = "minh863/fpt-chatbot-model"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN in .env file")
 
try:
    login(HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    logger.info(f"Loaded local model: {MODEL_ID}")
except Exception as e:
    logger.exception("Failed to load local model")
    raise e


# ===================== REQUEST & RESPONSE MODELS =====================

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    system_message: str = "You are an admissions consultant for FPT University. Please answer in a friendly, easy-to-understand, detailed manner about your major, tuition fees, extracurricular activities, and application process."
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    hf_token: str


class ChatResponse(BaseModel):
    response: str


# ===================== CHAT ENDPOINT =====================

@app.post("/api/chat", response_model=ChatResponse, summary="Generate chat response", tags=["chat"])
def chat_endpoint(req: ChatRequest): 
    if not req.hf_token:
        raise HTTPException(status_code=400, detail="hf_token is required")
     
    try:
        # Combine history into a single conversational context
        history_text = ""
        for turn in req.history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"

        # Construct prompt similar to system + conversation + user
        prompt = f"{req.system_message}\n{history_text}Học sinh: {req.message}\nTư vấn viên:"

        logger.debug("Prompt sent to local model:\n%s", prompt)

        output = pipe(
            prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
        )

        # Extract model response
        full_text = output[0]["generated_text"]
        logger.debug("Raw model output: %s", full_text)

        # Extract only the assistant's response after the last "Tư vấn viên:"
        response_text = full_text.split("Tư vấn viên:")[-1].strip()

    except Exception as e:
        logger.exception("Error during local model inference: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"response": response_text}


# ===================== MAIN =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
