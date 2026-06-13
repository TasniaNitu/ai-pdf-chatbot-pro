from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

print("Pages loaded:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # was 1200 — keeps ToC and explanatory sections intact
    chunk_overlap=300,    # was 200 — reduces mid-explanation splits
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

print("Chunks created:", len(chunks))

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("faiss_index")

print("FAISS index saved successfully!")