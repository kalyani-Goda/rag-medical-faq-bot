import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
dataset_path = os.getenv("DATASET_PATH")
# Load dataset (CSV with columns: Question, Answer, qtype)
documents = pd.read_csv(dataset_path)

answers = documents["Answer"].astype(str).tolist()
ids = [str(i) for i in range(len(answers))]
metadatas = documents[["qtype", "Question"]].to_dict("records")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

client = OpenAI(api_key=OPENAI_API_KEY)

class CustomEmbeddingFunction:
    def __call__(self, input):
        response = client.embeddings.create(
            model="text-embedding-3-small",  # or text-embedding-3-large
            input=input
        )
        return [item.embedding for item in response.data]

    def name(self):
        return "custom-openai-embedding"

embed_fn = CustomEmbeddingFunction()

# Create / get collection
DB_NAME = "medquaddb_openai"
db = chroma_client.get_or_create_collection(
    name=DB_NAME, embedding_function=embed_fn
)

# Add data in batches
batch_size = 100
for i in range(0, len(answers), batch_size):
    j = min(i + batch_size, len(answers))
    db.add(documents=answers[i:j], metadatas=metadatas[i:j], ids=ids[i:j])
    print(f"Added {i}..{j-1}")

print("Document count:", db.count())
