from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ==========================================
# LOAD EMBEDDINGS
# ==========================================

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# LOAD FAISS INDEX
# ==========================================

print("Loading FAISS index...")

try:
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS index loaded.")

except Exception as e:

    print(f"Error loading FAISS index: {e}")
    raise SystemExit(1)

# ==========================================
# CREATE RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# ==========================================
# LOAD OLLAMA MODEL
# ==========================================

print("Loading Ollama model...")

llm = OllamaLLM(
    model="llama3.2"
)

# ==========================================
# MEMORY
# ==========================================

chat_history = []

# ==========================================
# START CHATBOT
# ==========================================

print("\nChatbot ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("Ask a question: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    # ==========================================
    # RETRIEVE DOCUMENTS
    # ==========================================

    docs = retriever.invoke(question)

    if not docs:
        print("No relevant context found.")
        continue

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    context = context[:4000]

    # ==========================================
    # MEMORY
    # ==========================================

    history = "\n".join(
        chat_history[-6:]
    )

    # ==========================================
    # BUILD PROMPT
    # ==========================================

    prompt = f"""
You are a helpful assistant answering questions about a document.

Previous Conversation:
{history}

Document Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY using the document context.
- Do NOT use outside knowledge.
- If the answer is not in the context, reply exactly:
  I couldn't find that in the document.
- Keep answers concise and accurate.

Answer:
"""

    print("\nGenerating answer...\n")

    try:

        answer = llm.invoke(prompt).strip()

    except Exception as e:

        print(f"\nError generating answer:\n{e}")
        continue

    # ==========================================
    # SAVE MEMORY
    # ==========================================

    chat_history.append(
        f"User: {question}"
    )

    chat_history.append(
        f"Assistant: {answer}"
    )

    if len(chat_history) > 12:
        chat_history = chat_history[-12:]

    # ==========================================
    # DISPLAY ANSWER
    # ==========================================

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)
    print()