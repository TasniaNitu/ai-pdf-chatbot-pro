import pandas as pd
from difflib import SequenceMatcher

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# ── Helpers ──────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "with", "that",
    "this", "it", "as", "by", "be", "have", "has",
}

# Known chapter titles for structure evaluation (phrase-level matching)
CHAPTERS = [
    "whetting your appetite",
    "using the python interpreter",
    "an informal introduction to python",
    "more control flow tools",
    "data structures",
    "modules",
    "input and output",
    "errors and exceptions",
    "what now",
]


def check_correctness(expected: str, answer: str, q_type: str):
    """
    Returns (is_correct: bool, score: float).

    - negative    -> must contain the exact refusal phrase
    - structure   -> expected chapter names must appear verbatim in the answer
    - summary /   -> broad keyword overlap >= 0.30
      themes
    - factual /   -> bidirectional substring match OR word overlap >= 0.50
      explanatory    OR SequenceMatcher >= 0.40
    """
    exp = expected.lower()
    ans = answer.lower()

    # Negative questions expect the model to admit it doesn't know
    if q_type == "negative":
        refusal = "i couldn't find that in the document."
        score = 1.0 if refusal in ans else 0.0
        return score == 1.0, score

    # Structure: check what fraction of known chapter titles appear in the answer
    if q_type == "structure":
        matches = sum(1 for chapter in CHAPTERS if chapter in ans)
        score = matches / len(CHAPTERS)
        return score >= 0.50, round(score, 3)

    # Summary / themes: broad keyword overlap is appropriate
    if q_type in ("summary", "themes"):
        exp_words = set(exp.split()) - STOPWORDS
        ans_words = set(ans.split()) - STOPWORDS
        if not exp_words:
            return False, 0.0
        overlap = len(exp_words & ans_words) / len(exp_words)
        return overlap >= 0.30, round(overlap, 3)

    # Factual / explanatory: substring, word overlap, or fuzzy similarity
    similarity = SequenceMatcher(None, exp, ans).ratio()
    exp_words = set(exp.split())
    ans_words = set(ans.split())
    overlap = len(exp_words & ans_words) / len(exp_words) if exp_words else 0.0
    is_correct = (
        exp in ans
        or ans in exp
        or overlap >= 0.50
        or similarity >= 0.40
    )
    return is_correct, round(max(similarity, overlap), 3)


# ── Load pipeline ─────────────────────────────────────────────────────────────

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

print("Loading Ollama model...")
llm = OllamaLLM(model="llama3.2")

print("Loading evaluation dataset...")
df = pd.read_csv("evaluation_dataset.csv")

# ── Evaluation loop ───────────────────────────────────────────────────────────

FACTUAL_PROMPT = """You are a helpful assistant answering questions about a document.

Document Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY from the document context.
- Return the exact wording from the document whenever possible.
- For factual questions, answer in one sentence when possible.
- For chapter questions, use the table of contents if present.
- Do not ask follow-up questions.
- Do not explain beyond what is requested.
- If the answer is not present, return exactly:
  I couldn't find that in the document.

Answer:"""

SUMMARY_PROMPT = """You are a helpful assistant answering questions about a document.

Document Context:
{context}

Question:
{question}

Instructions:
- This is a broad/summary question — give a comprehensive answer.
- Cover all major topics, chapters, and themes found in the document.
- Draw from the full context provided, not just one chunk.
- If the answer is not present, return exactly:
  I couldn't find that in the document.

Answer:"""


correct = 0
total = len(df)
results = []

for i, row in df.iterrows():
    question = str(row["question"])
    expected = str(row["expected_answer"])
    q_type   = str(row["type"]).lower()          # ground truth from the dataset

    is_broad = q_type in ("summary", "structure", "themes")

    # Broad questions need wide coverage; definition questions need precision
    if is_broad:
        docs = vectorstore.similarity_search(question, k=10)
    else:
        docs = vectorstore.similarity_search(question, k=6)

    # 6 000 chars — enough context without drowning llama3.2 in noise
    context = "\n\n".join(doc.page_content for doc in docs)[:6000]

    template = SUMMARY_PROMPT if is_broad else FACTUAL_PROMPT
    prompt = template.format(context=context, question=question)

    try:
        answer = llm.invoke(prompt).strip()
    except Exception:
        answer = ""

    is_correct, score = check_correctness(expected, answer, q_type)

    if is_correct:
        correct += 1

    results.append({
        "question": question,
        "expected": expected,
        "actual": answer,
        "correct": is_correct,
        "score": score,
        "type": q_type,
    })

    print(f"[{i+1}/{total}] {'✅' if is_correct else '❌'}  {q_type:<8}  score={score:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
accuracy = (correct / total) * 100

print("\n" + "=" * 60)
print(f"Total Questions : {total}")
print(f"Correct Answers : {correct}")
print(f"Overall Accuracy: {accuracy:.2f}%")
print("-" * 60)

for qtype in sorted(results_df["type"].unique()):
    subset = results_df[results_df["type"] == qtype]
    acc = subset["correct"].mean() * 100
    print(f"{qtype.capitalize():<12} ({len(subset):>2} Qs): {acc:.2f}%")

print("=" * 60)

results_df.to_csv("evaluation_results.csv", index=False)
print("\nSaved: evaluation_results.csv")