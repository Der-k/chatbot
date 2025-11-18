import os
import pickle
import time
from flask import Flask, render_template, request, jsonify # <-- Import Flask components
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Flask App
app = Flask(__name__) # <-- Flask Initialization

# === Paths ===
DATA_FOLDER = "data"
DB_PATH = "vectorstore.faiss"
META_PATH = "vectorstore_meta.pkl"

# Ensure data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"⚠️ Created missing '{DATA_FOLDER}' folder. Add .txt files here for RAG data.")


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
        # Added a check for file existence before loading to prevent startup errors if paths are wrong
        if os.path.exists(DB_PATH):
            return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            print("⚠️ FAISS file missing, attempting to rebuild...")

    print("🔹 Rebuilding FAISS database (changes detected or missing files)...")
    documents = load_documents()
    if not documents:
        print("🛑 No text documents found in 'data/' folder. Please add some .txt files.")
        # Return a dummy FAISS store or raise an error, depending on desired behavior
        # For simplicity, we'll try to continue, but the RAG will fail if no docs.
        # This is a good place for more robust error handling.
        return None 

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
if db:
    retriever = db.as_retriever()
else:
    retriever = None # Handle the case where the DB couldn't be built


# === Use Groq’s free cloud LLM ===
# IMPORTANT: Never hardcode your API key in production code. Use os.getenv()
# For this example, we'll use the environment variable
os.environ["GROQ_API_KEY"] = "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" # Replace this with a proper environment variable setup
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"🛑 Could not initialize ChatGroq LLM. Check API key/internet connection. Error: {e}")
    llm = None


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
    if not retriever or not llm:
        return "System error: The RAG service is currently unavailable. Please check the logs."

    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        print(f"An error occurred during retrieval/LLM chain: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again."


# === Flask Routes ===

# 1. Main Chat Interface
@app.route("/")
def index():
    """Renders the main chat interface page."""
    # Pass the company name to the template
    return render_template("index.html", company_name="CanvassingTech")

# 2. API Endpoint for Chat Messages
@app.route("/ask", methods=["POST"])
def ask_bot_api():
    """Handles the AJAX request from the frontend with the user's question."""
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"response": "Please provide a question."}), 400

    # Get the answer from your RAG logic
    bot_response = retrieve_and_answer(user_question)

    # Return the response as JSON
    return jsonify({"response": bot_response})


# === Run the Flask App ===
if __name__ == "__main__":
    print("\n🚀 Starting Flask Web App...")
    # Setting host='0.0.0.0' makes it accessible externally if needed (e.g., in a container)
    app.run(host="0.0.0.0", port=5000, debug=True)