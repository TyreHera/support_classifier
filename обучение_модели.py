import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import pymorphy3

nltk.download('stopwords', quiet=True)
russian_stop_words = stopwords.words("russian")
morph = pymorphy3.MorphAnalyzer()

def clean_and_lemmatize(text):
    text = str(text).lower()
    words = re.findall(r'[а-яёa-z]+', text)
    lemmas = []
    for word in words:
        if word not in russian_stop_words:
            lemma = morph.parse(word)[0].normal_form
            lemmas.append(lemma)
            
    return " ".join(lemmas)

print("Загрузка и предобработка данных (это может занять пару секунд)...")
df = pd.read_csv("dataset.csv", sep=";", encoding="utf-8")

df['text_clean'] = df['text'].apply(clean_and_lemmatize)
df['target_combined'] = df['category'] + "_" + df['priority']

X = df['text_clean']
y = df['target_combined']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_combined = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42))])

print("Обучаем модель...")
model_combined.fit(X_train, y_train)

y_pred = model_combined.predict(X_test)
print(f"Accuracy (Категория + Приоритет): {accuracy_score(y_test, y_pred):.2f}")

joblib.dump(model_combined, "model_combined.joblib")
print("Единая модель успешно сохранена!")
