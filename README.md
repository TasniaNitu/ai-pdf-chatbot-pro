# AI PDF Chatbot Pro

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-005571?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)

A Retrieval-Augmented Generation application that allows users to upload PDF documents and ask natural-language questions about their contents.

The system retrieves relevant document passages using FAISS and generates context-aware answers with a locally running Llama 3.2 model through Ollama.

![AI PDF Chatbot Pro Demo](demo.gif)

---

## Demo

The demonstration above shows:

- Uploading and processing a PDF
- Asking factual questions
- Retrieving relevant source passages
- Generating answers with page references
- Declining questions when supporting information is not found in the document

---

## Project Overview

AI PDF Chatbot Pro is a document question-answering system built with Retrieval-Augmented Generation.

Uploaded PDFs are:

1. Extracted into text
2. Divided into overlapping chunks
3. Converted into vector embeddings
4. Indexed in a FAISS vector store
5. Retrieved according to semantic relevance
6. Passed to a locally running Llama 3.2 model as supporting context

The application is designed to produce retrieval-grounded responses rather than relying only on the language model’s internal knowledge.

---

## Architecture

```text
Uploaded PDF
     ↓
Text extraction
     ↓
Recursive text chunking
     ↓
Hugging Face sentence embeddings
     ↓
FAISS vector index
     ↓
Semantic retrieval
     ↓
Context-enriched prompt
     ↓
Llama 3.2 through Ollama
     ↓
Answer with source-page reference
```

---

## Features

- Upload PDF documents through a Streamlit interface
- Extract and divide PDF text into searchable chunks
- Perform semantic search using FAISS
- Generate answers with an Ollama-powered local LLM
- Display source-page references
- Preserve recent conversation history
- Adjust the retrieval similarity threshold
- Clear chat history or reset the active document
- Decline unsupported out-of-document questions
- Evaluate responses using a reproducible question–answer dataset

---

## Tech Stack

| Area | Technology |
|---|---|
| Programming language | Python |
| Interface | Streamlit |
| RAG orchestration | LangChain |
| Vector search | FAISS |
| Embeddings | Hugging Face sentence-transformer embeddings |
| Local inference | Ollama |
| Language model | Llama 3.2 |
| PDF processing | LangChain PDF loader |
| Evaluation | Custom Python evaluation pipeline |

---

## Evaluation

The project was evaluated on a small, document-specific dataset containing **20 curated question–answer pairs** based on the included sample PDF.

| Metric | Result |
|---|---:|
| Total questions | 20 |
| Correct answers | 18 |
| Overall answer accuracy | 90% |

### Category results

| Category | Accuracy |
|---|---:|
| Factual | 100% |
| Explanatory | 100% |
| Negative / out-of-document | 100% |
| Themes | 100% |
| Summary | 50% |
| Document structure | 0% |

The results show strong performance on factual retrieval and unsupported-question handling. Summary and document-structure questions remain the main areas for improvement.

Run the evaluation with:

```bash
python evaluate.py
```

The generated results are stored in:

```text
evaluation_results.csv
```

### Evaluation limitations

- The benchmark contains only 20 questions.
- All questions are based on one sample document.
- The result does not represent performance on every PDF type or subject.
- Answer correctness is determined using the project’s custom evaluation procedure.
- Scanned PDFs, tables, diagrams, and complex document layouts may require additional processing.
- A larger human-reviewed benchmark would provide stronger evidence of general performance.

---

## Project Structure

```text
ai-pdf-chatbot-pro/
├── app.py
├── ingest.py
├── query.py
├── evaluate.py
├── evaluation_dataset.csv
├── evaluation_results.csv
├── demo.gif
├── requirements.txt
├── README.md
└── data/
    └── sample.pdf
```

### Main files

- `app.py` — Streamlit user interface and application flow
- `ingest.py` — PDF loading, chunking, embeddings, and FAISS indexing
- `query.py` — Retrieval and answer-generation pipeline
- `evaluate.py` — Evaluation workflow
- `evaluation_dataset.csv` — Curated question–answer dataset
- `evaluation_results.csv` — Saved evaluation output
- `data/sample.pdf` — Sample document used for demonstration and testing

---

## Local Setup

### Prerequisites

- Python 3.10 or later
- Ollama installed locally
- Sufficient storage and memory for the selected Ollama model

### 1. Clone the repository

```bash
git clone https://github.com/TasniaNitu/ai-pdf-chatbot-pro.git
cd ai-pdf-chatbot-pro
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Download Ollama from its official website, start the Ollama service, and pull the model:

```bash
ollama pull llama3.2
```

### 5. Run the application

```bash
streamlit run app.py
```

The application should open in the default browser.

---

## Usage

1. Upload a PDF using the sidebar.
2. Wait for document processing to complete.
3. Enter a question about the uploaded document.
4. Review the generated answer and source-page reference.
5. Ask follow-up questions using the same document context.
6. Use **Clear Chat** or **Reset PDF** when necessary.

---

## Example Questions

For a general document:

- What are the main topics covered?
- What conclusions does the document present?
- Explain the section about a specific topic.
- List the key points from the document.
- Summarize the document.

For the included Python tutorial sample:

- What is a tuple?
- What is a dictionary?
- What is a set?
- What is a class?
- What are the major topics covered?

---

## Limitations

- The current implementation is designed primarily for text-based PDFs.
- Scanned or image-only PDFs may not work because OCR is not included.
- Summary and document-structure questions are less reliable than factual questions.
- Retrieval quality depends on document formatting, chunking, and wording.
- The application runs locally because it depends on Ollama.
- Generated answers should be checked against the cited source passages.
- Uploaded sensitive documents should be handled according to appropriate privacy and security requirements.

---

## Future Improvements

- Better chapter- and section-aware chunking
- Hybrid semantic and keyword retrieval
- Reranking of retrieved passages
- Improved long-document summarization
- OCR support for scanned PDFs
- Table and figure extraction
- Larger human-reviewed evaluation dataset
- Retrieval metrics such as Recall@k and Mean Reciprocal Rank
- Additional local model options
- Public deployment using a compatible hosted inference service

---

## Author

**Kazi Tasnia Nitu**
- [GitHub profile](https://github.com/TasniaNitu)
- [Portfolio](https://tasnianitu.github.io)
- [LinkedIn](https://www.linkedin.com/in/tasnia-ai)
