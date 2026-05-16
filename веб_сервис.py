from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="Классификатор обращений API")

# Загружаем сохраненные модели
try:
    model_category = joblib.load("model_category.joblib")
    model_priority = joblib.load("model_priority.joblib")
except FileNotFoundError:
    raise RuntimeError("Файлы моделей не найдены. Сначала запустите train_model.py")

# Pydantic схема для входящего JSON [cite: 63]
class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_ticket(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст обращения не может быть пустым")
    
    # Делаем предсказание
    category = model_category.predict([request.text])[0]
    priority = model_priority.predict([request.text])[0]
    
    # Возвращаем корректный JSON [cite: 78]
    return {
        "text": request.text,
        "category": category,
        "priority": priority
    }

# Запуск: uvicorn api:app --reload