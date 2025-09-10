# 🩺 RAG-based Medical FAQ Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers patient medical FAQs using OpenAI LLMs and a vector database (ChromaDB).  
It was built as part of the **AI/ML Engineer Assignment – RAG Systems**.

---

## 🚀 Features
- Preprocesses and embeds medical FAQ dataset (`medical_faq.csv`)  
- Stores embeddings in **ChromaDB** (local, free, no cloud required)  
- Retrieves the most relevant passages for user queries  
- Uses **OpenAI GPT** (free credits) to generate clear, natural answers  
- Provides a simple **Streamlit interface** for interactive Q&A  

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **ChromaDB** (vector store)
- **OpenAI API** (`text-embedding-3-small`, `gpt-4o-mini` or `gpt-3.5-turbo`)
- **Streamlit** (chat UI)

---

## 📦 Setup

1. **Clone repo**
   ```bash
   git clone https://github.com/your-username/rag-medical-faq-bot.git
   cd rag-medical-faq-bot

2. **Install dependencies**
     ```bash
     pip install -r requirements.txt
3. **Set OpenAI key**
    ```bash
    export OPENAI_API_KEY="your_api_key"

## ▶️ Usage

1. **Build Index (Embeddings + Chroma)**
    ```bash
    python src/build_index.py
2. **Run Chatbot (Command Line)**
    ```bash
    python src/chatbot.py
    Example:

    text
    Q: What are the early symptoms of diabetes?
    A: Some early symptoms of diabetes include frequent urination, excessive thirst, constant hunger, and fatigue...
3. **Run Streamlit UI**
    ```bash
    streamlit run src/ui_streamlit.py
    Then open http://localhost:8501 in your browser.

## 📂 File Structure
    text
    ├── src/
    │   ├── build_index.py      # Preprocesses dataset, creates embeddings, stores in ChromaDB
    │   ├── chatbot.py          # Retrieves top-k passages and generates response
    │   └── ui_streamlit.py     # Web interface for patient queries
    ├── data/
    │   └── medical_faq.csv     # Sample dataset
    ├── requirements.txt        # Python dependencies
    └── README.md              # This file

## 🧠 Design Choices

**Embeddings:**
    Used text-embedding-3-small for cost efficiency and high performance on retrieval tasks.

**Vector Store:**
    ChromaDB (open-source, simple, free).

**LLM:**
    Used OpenAI GPT (gpt-4o-mini or gpt-3.5-turbo) for natural, conversational answers.

**Chunking:**
    Long answers split into smaller passages (~200 tokens) for better retrieval.

**Top-k Retrieval:**
    Query retrieves top-3 passages, concatenated into prompt.

**UI:**
    Streamlit app for quick demo + CLI fallback.

## ✅ Example Queries
    "What are the early symptoms of diabetes?"

    "Can children take paracetamol?"

    "What foods are good for heart health?"

    "How to manage high blood pressure?"

    "What is the recommended daily water intake?"


## 📽️ Demo(Optional)
    [Add screenshot or video demo link here]