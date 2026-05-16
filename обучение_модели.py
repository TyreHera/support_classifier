import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. Загрузка данных
df = pd.read_csv("dataset.csv", sep=";", encoding="utf-8")

# 2. Разделение на тренировочную и тестовую выборки
X = df['text']
y_cat = df['category']
y_prio = df['priority']

X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
    X, y_cat, y_prio, test_size=0.2, random_state=42
)

# 3. Создание пайплайнов (Векторизация текста + Логистическая регрессия)
# Пайплайн для категории
model_category = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True)),
    ('clf', LogisticRegression())
])

# Пайплайн для приоритета
model_priority = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True)),
    ('clf', LogisticRegression())
])

# 4. Обучение моделей
print("Обучаем модели...")
model_category.fit(X_train, y_cat_train)
model_priority.fit(X_train, y_prio_train)

# 5. Оценка качества (как требует кейс)
cat_pred = model_category.predict(X_test)
prio_pred = model_priority.predict(X_test)

print(f"Accuracy категории: {accuracy_score(y_cat_test, cat_pred):.2f}")
print(f"Accuracy приоритета: {accuracy_score(y_prio_test, prio_pred):.2f}")

# 6. Сохранение моделей в файлы
joblib.dump(model_category, "model_category.joblib")
joblib.dump(model_priority, "model_priority.joblib")
print("Модели успешно сохранены!")
