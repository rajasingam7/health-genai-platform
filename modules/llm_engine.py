import streamlit as st
from groq import Groq


def get_groq_client():
    """
    Initialize Groq client using Streamlit secrets.
    """

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not configured in Streamlit secrets.")
        st.stop()

    return Groq(api_key=api_key)


def generate_explanation(analytics_result: dict, user_question: str) -> str:
    client = get_groq_client()

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a professional healthcare analytics assistant. Provide structured and concise explanations."
            },
            {
                "role": "user",
                "content": f"Question: {user_question}\nData: {analytics_result}"
            }
        ],
        temperature=0.3,
        max_tokens=400
    )

    return completion.choices[0].message.content