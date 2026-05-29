import streamlit as st
import requests

st.title("Служба поддержки: Умный классификатор")
st.write("Введите текст вашего обращения, и наш ML-сервис автоматически определит его категорию и приоритет.")

user_input = st.text_area("Текст обращения:", placeholder="Например: У меня не проходит оплата")

if st.button("Отправить"):
    if user_input:
        try:
            response = requests.post("http://localhost:8000/predict", json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                st.success("Обращение успешно обработано!")
                col1, col2 = st.columns(2)
                col1.metric("Категория", result["category"])
                col2.metric("Приоритет", result["priority"])
            else:
                st.error(f"Ошибка API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к серверу API. Убедитесь, что api.py запущен (uvicorn api:app --reload).")
    else:
        st.warning("Пожалуйста, введите текст обращения.")
