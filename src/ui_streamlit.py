import streamlit as st
from chatbot import generate_answer

st.set_page_config(page_title="MedQuad QA Chatbot", layout="centered")

st.title("💬 MedQuad Chatbot")
st.write("Ask me questions, and I'll answer based on the knowledge base.")

# Input box
user_query = st.text_input("Enter your question:")

if st.button("Ask") and user_query:
    with st.spinner("Thinking..."):
        answer = generate_answer(user_query)
    st.success("Answer:")
    st.write(answer)
