# AI PDF Chatbot Pro

![Demo](demo.gif)

## Demo

The GIF above demonstrates:

- Uploading a PDF
- Asking factual questions
- Answer generation with source retrieval
- Hallucination-resistant responses for out-of-document questions

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask natural-language questions about their contents.

## Project Overview

AI PDF Chatbot Pro is a Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask questions about their contents. The system uses FAISS vector search, Hugging Face embeddings, and the Ollama-powered Llama 3.2 model to retrieve relevant document chunks and generate context-aware answers.

The project includes an evaluation framework using a custom dataset of 20 question–answer pairs, achieving 90% overall answer accuracy.

Uploaded PDFs are split into text chunks, embedded using Hugging Face sentence-transformer embeddings, indexed with FAISS, and retrieved to provide context for the Llama 3.2 model.

## Features

* Upload PDF documents
* Semantic search using FAISS vector database
* Local LLM inference using Ollama (llama3.2)
* Source attribution with page references
* Conversation history
* Evaluation framework with accuracy reporting
* Hallucination-resistant responses

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face Embeddings
* Ollama
* Llama 3.2

## Evaluation Results

Evaluation was performed using a custom dataset of 20 question-answer pairs.

| Metric           | Result |
| ---------------- | ------ |
| Total Questions  | 20     |
| Correct Answers  | 18     |
| Overall Accuracy | 90%    |

Category Results:

* Factual: 100%
* Explanatory: 100%
* Negative: 100%
* Themes: 100%
* Summary: 50%
* Structure: 0%

## Project Structure

- `app.py` – Streamlit application
- `ingest.py` – PDF processing and vector indexing
- `query.py` – Question-answering pipeline
- `evaluate.py` – Evaluation framework
- `evaluation_dataset.csv` – Evaluation dataset
- `evaluation_results.csv` – Evaluation results
- `data/sample.pdf` – Sample document used for testing and evaluation

## Installation

Install Ollama from:

https://ollama.com

Clone the repository:

```bash
git clone https://github.com/TasniaNitu/ai-pdf-chatbot-pro.git
cd ai-pdf-chatbot-pro
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the model:

```bash
ollama pull llama3.2
```

Run the application:

```bash
streamlit run app.py
```

## Deployment Notes

This project uses Ollama for local LLM inference and is designed to run locally on the user's machine.

To use the chatbot:

1. Install Ollama
2. Download the Llama 3.2 model:

   ```bash
   ollama pull llama3.2
   ```
3. Start Ollama
4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

A demo GIF demonstrating the full workflow is included at the top of this repository.

## Example Questions

After uploading a PDF, users can ask questions such as:

- Summarize this document.
- What are the main topics covered?
- Explain Chapter 3.
- What conclusions does the document present?
- What does the document say about [topic]?
- List the key points from this PDF.

## Sample PDF

This repository includes a sample PDF (`data/sample.pdf`), based on the Python Tutorial, which is used for testing and evaluating the chatbot.

If you use the sample PDF, you can try questions such as:

* What is a tuple?
* What is a dictionary?
* What is a set?
* What is a class?
* What are the major topics covered?
* Summarize the document.

## Future Improvements

* Better chapter-level retrieval
* Improved summary generation
* Public cloud deployment
* Hybrid retrieval methods

## Author

Tasnia Nitu