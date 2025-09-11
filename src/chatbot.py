
import chromadb
import requests
import json
import os
import shutil
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_PATH = os.getenv("DB_PATH")
googlekey = os.getenv("GOOGLE_API_KEY")

# Load DB & embedding model
chroma_client = chromadb.PersistentClient(path=DB_PATH)
db = chroma_client.get_collection("medical_faqs")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=googlekey)

def retrieve_context(query, n_results=3):
    query_emb = embedding_model.encode(query).tolist()
    results = db.query(query_embeddings=[query_emb], n_results=n_results)
    return results["documents"][0]


def generate_prompt(query: str):
    # Build prompt for the LLM
    passages = retrieve_context(query)
    prompt = "You are a helpful assistant. Use the PASSAGEs below as a context to answer the QUESTION. If passages don't answer, say you don't have enough info. However, you are talking to a non-medical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone. Be sure to respond in a complete sentence with relavent informtion.\n\n"
    prompt += f"QUESTION: {query}\n\n"
    for idx, p in enumerate(passages):
        prompt += f"PASSAGE {idx+1}: {p}\n\n"
    prompt += "Answer (concise, friendly):"
    return prompt

def generate_answer(query: str):

    prompt = generate_prompt(query)
    print(prompt)

    answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)

    return answer.text

if __name__ == "__main__":
    # Example query
    user_query = "What are the symptoms of diabetes?"

    print("🔎 Query:", user_query)
    answer = generate_answer(user_query)
    print("\n💡 Answer:\n", answer)

