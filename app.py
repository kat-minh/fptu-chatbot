from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI(
    title="Chat API",
    version="0.1.0",
    description="An API for a friendly chatbot using Hugging Face InferenceClient. Swagger UI is available at /swagger.",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

MODEL_ID = "minh863/fpt-chatbot-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)


SYSTEM_PROMPT = (
    "Bạn là tư vấn viên tuyển sinh của Trường Đại học FPT. "
    "Hãy trả lời thân thiện, dễ hiểu, chi tiết về ngành học, học phí, hoạt động ngoại khóa, quy trình đăng ký."
)

@app.post("/api/chat")
async def predict(data: dict):
    question = data.get("text", "")
    prompt = f"{SYSTEM_PROMPT}\nHọc sinh: {question}\nTư vấn viên:"
    output = pipe(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True)
    answer = output[0]["generated_text"].split("Tư vấn viên:")[-1].strip()
    return {"reply": answer}
  
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
