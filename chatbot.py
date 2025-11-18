import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# === Paths ===
DATA_FOLDER = "data"
DB_PATH = "vectorstore.faiss"
META_PATH = "vectorstore_meta.pkl"


# === Helper: Check if any file changed ===
def data_files_changed():
    """Check if any .txt file in data/ is newer than the FAISS build."""
    if not os.path.exists(META_PATH):
        return True

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    last_build = meta.get("timestamp", 0)

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, file)
            if os.path.getmtime(file_path) > last_build:
                return True
    return False


# === Load and split documents ===
def load_documents():
    all_docs = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_FOLDER, file))
            all_docs.extend(loader.load())
    return all_docs


# === Initialize embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# === Load or Rebuild Vector Database ===
def load_or_build_db():
    if os.path.exists(DB_PATH) and os.path.exists(META_PATH) and not data_files_changed():
        print("🔹 Loading existing FAISS database (no file changes detected)...")
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    print("🔹 Rebuilding FAISS database (changes detected or missing files)...")
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)

    # Save FAISS + metadata
    db.save_local(DB_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"timestamp": time.time()}, f)

    print("✅ Vector database built and saved.")
    return db


db = load_or_build_db()
retriever = db.as_retriever()


# === Use Groq’s free cloud LLM ===
os.environ["GROQ_API_KEY"] = "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"  # https://console.groq.com/keys
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("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"))



prompt = ChatPromptTemplate.from_template(
    """
You are the official support assistant for CanvassingTech.

### Response Rules:
- Keep responses **short and clear** (2–5 sentences maximum).
- Use **only** the information provided in the context.
- Do **not** guess or invent information. If the answer is not in the context, say:
  "I don’t have that information available at the moment."
- Maintain a **professional, corporate** tone.
- Answer the question **directly and factually**.
- No filler, no long explanations, no repeated statements.

---

**User Question:** {question}

**Relevant Context:**  
{context}

---

**Answer:**
"""
)



# === Retrieval + LLM Chain ===
def retrieve_and_answer(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    return response.content


def ask_bot(question: str) -> str:
    return retrieve_and_answer(question)


# === Main loop ===
if __name__ == "__main__":
    print("\n✅ Chatbot is ready! Type your question below.\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Bot:", ask_bot(q))
