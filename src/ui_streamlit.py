import streamlit as st
from chatbot import generate_answer, retrieve_context
import time
st.title("🩺 Medical FAQ Chatbot (RAG + Ollama)")

query = st.text_input("Ask a medical question:")

if st.button("Ask") and query:
    start = time.perf_counter()  # ⏱ start time
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
        context = retrieve_context(query)
    end = time.perf_counter()    # ⏱ end time
    runtime = end - start

    st.subheader("Answer")
    st.write(answer)
    st.write(f"⏱ Runtime: {runtime:.2f} seconds")
    with st.expander("Retrieved Context"):
        st.write(context)
