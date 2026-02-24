import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

# Load .env for local development
load_dotenv()


def get_groq_client():
    """
    Initialize Groq client.
    Works for:
    - Local (.env)
    - Streamlit Cloud (secrets.toml)
    """

    api_key = None

    # 1️⃣ Try Streamlit Cloud secrets safely
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None

    # 2️⃣ Fallback to local .env
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    # 3️⃣ Final validation
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not configured.")
        st.info("Set GROQ_API_KEY in .env (local) or secrets.toml (cloud).")
        st.stop()

    return Groq(api_key=api_key)


def generate_explanation(analytics_result: dict, user_question: str) -> str:
    """
    Generate explanation from structured analytics or prediction output.
    Only structured aggregated data is sent to the LLM.
    """

    client = get_groq_client()

    system_prompt = """
    You are a professional healthcare analytics assistant.
    Provide structured, concise, and professional explanations.
    Do not hallucinate patient-level data.
    Avoid giving direct medical prescriptions.
    Base your explanation strictly on the provided structured data.
    """

    user_prompt = f"""
    User Question:
    {user_question}

    Structured Data:
    {analytics_result}

    Provide a clear professional explanation.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ Supported Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"LLM generation failed: {str(e)}"