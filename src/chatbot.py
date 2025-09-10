import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define same embedding function used in build_index.py
class CustomEmbeddingFunction:
    def __call__(self, input):
        response = client.embeddings.create(
            model="text-embedding-3-small",  # must match build_index.py
            input=input
        )
        return [item.embedding for item in response.data]

    def name(self):
        return "custom-openai-embedding"

# Initialize Chroma with persistent DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Load collection with the same embedding function
DB_NAME = "medquaddb_openai"
embed_fn = CustomEmbeddingFunction()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Functions
def retrieve_context(query: str, n_results: int = 3):
    results = db.query(query_texts=[query], n_results=n_results)
    return results

def generate_answer(query: str):
    results = retrieve_context(query)
    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)
    prompt = f"""
    You are a helpful medical assistant.
    Use the following context to answer the question.

    Question: {query}
    Context: {context}
    """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content
