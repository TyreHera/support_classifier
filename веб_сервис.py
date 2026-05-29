from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import pymorphy3
import nltk
from nltk.corpus import stopwords

app = FastAPI(title="Классификатор обращений API")

nltk.download('stopwords', quiet=True)
russian_stop_words = stopwords.words("russian")
morph = pymorphy3.MorphAnalyzer()

def clean_and_lemmatize(text):
    words = re.findall(r'[а-яёa-z]+', str(text).lower())
    lemmas = [morph.parse(w)[0].normal_form for w in words if w not in russian_stop_words]
    return " ".join(lemmas)

try:
    model_combined = joblib.load("model_combined.joblib")
except FileNotFoundError:
    raise RuntimeError("Файл модели не найден. Сначала запустите обучение_модели.py")

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_ticket(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст обращения не может быть пустым")
    processed_text = clean_and_lemmatize(request.text)
    prediction = model_combined.predict([processed_text])[0]
    category, priority = prediction.split("_")
    
    return {
        "text": request.text,
        "category": category,
        "priority": priority
    }
