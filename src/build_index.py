import os
import shutil
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_PATH = os.getenv("DB_PATH")

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("Old ChromaDB database deleted!")

DATASET_PATH = os.path.join("Data", "medical_faq.csv")

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    return df

# Load dataset (CSV with columns: Question, Answer, qtype)
documents = load_data(DATASET_PATH)

# Embedding model (local)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = (documents["Question"] + " " + documents["Answer"]).astype(str).tolist()
answers = documents["Answer"].astype(str).tolist()
ids = [str(i) for i in range(len(texts))]
metadatas = documents[["qtype", "Question"]].to_dict("records")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection("medical_faqs")

# Convert to tensor first, then to numpy manually
embeddings_tensor = embedding_model.encode(texts, convert_to_tensor=True)
embeddings = embeddings_tensor.cpu().numpy().tolist()

# Use a smaller batch size to be safe
batch_size = 2500
total_items = len(ids)

for i in range(0, total_items, batch_size):
    end_idx = min(i + batch_size, total_items)
    
    batch_ids = ids[i:end_idx]
    batch_documents = texts[i:end_idx]
    batch_embeddings = embeddings[i:end_idx]
    batch_metadatas = metadatas[i:end_idx]
    
    print(f"Adding batch {i//batch_size + 1}: items {i} to {end_idx-1}")
    
    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

print("Index built successfully!")