import streamlit as st
import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.3-2b-instruct"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
# Function to query Hugging Face Inference API
def query_model(prompt, max_tokens=300, temperature=0.7):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"]
    else:
        return f"⚠️ API Error: {response.status_code} - {response.text}"

# Streamlit layout
st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🧠 HealthAI: Intelligent Healthcare Assistant")

menu = st.sidebar.selectbox("Choose a Module", [
    "👨‍💎 Patient Chat",
    "🔍 Disease Prediction",
    "💊 Treatment Plans",
    "📊 Health Analytics"
])

st.sidebar.markdown("---")
st.sidebar.caption("Powered by IBM Granite · via Hugging Face")

# 1. Patient Chat
if menu == "👨‍💎 Patient Chat":
    st.header("💬 Ask a Health Question")
    question = st.text_input("What would you like to ask?")
    if st.button("Get Answer") and question.strip():
        prompt = f"User: {question}\nHealthAI:"
        with st.spinner("Thinking..."):
            result = query_model(prompt)
            response = result.split("HealthAI:")[-1].strip()
        st.success(response)
        st.caption("⚠️ This is for informational use only.")

# 2. Disease Prediction
elif menu == "🔍 Disease Prediction":
    st.header("🩺 Symptom-Based Disease Prediction")
    symptoms = st.text_area("Enter your symptoms (comma separated)")
    age = st.slider("Age", 0, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])

    if st.button("Predict Disease") and symptoms:
        prompt = (
            f"A patient reports the following symptoms: {symptoms}. "
            f"Age: {age}, Gender: {gender}. "
            f"Based on this, what are the top 3 likely conditions with brief explanations?"
        )
        with st.spinner("Analyzing..."):
            result = query_model(prompt, max_tokens=400, temperature=0.8)
            response = result.split("conditions")[-1].strip()
        st.success("🔍 Predicted Conditions:")
        st.write(response)
        st.caption("📌 Consult a doctor for confirmation.")

# 3. Treatment Plans
elif menu == "💊 Treatment Plans":
    st.header("📋 Treatment Plan Generator")
    condition = st.text_input("Enter a diagnosed condition")
    if st.button("Generate Plan") and condition:
        prompt = (
            f"Create a detailed treatment plan for {condition}. "
            f"Include medication, lifestyle changes, and recommended tests."
        )
        with st.spinner("Generating plan..."):
            result = query_model(prompt, max_tokens=400)
            response = result.split("plan")[-1].strip()
        st.success("🧾 Suggested Treatment:")
        st.write(response)

# 4. Health Analytics
elif menu == "📊 Health Analytics":
    st.header("📈 Upload & Analyze Health Data")
    uploaded_file = st.file_uploader("Upload your health data CSV (date, heart_rate, bp, glucose)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        df.set_index("date", inplace=True)

        st.subheader("🧪 Metrics Summary")
        metric = st.selectbox("Select a metric to visualize", df.columns)
        st.line_chart(df[metric])
        st.write(df[metric].describe())

        # AI Insight
        prompt = f"Analyze this time series health data: {df[metric].tolist()[:20]}. What patterns or risks do you notice?"
        with st.spinner("Analyzing trend..."):
            result = query_model(prompt, max_tokens=200, temperature=0.6)
            insight = result.split("data:")[-1].strip()
        st.success("🧠 AI Insight:")
        st.write(insight)

# Footer
st.markdown("---")
st.caption("© 2025 HealthAI · Streamlit + Hugging Face + IBM Granite")