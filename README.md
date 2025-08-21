# Financial Policy Chatbot

A simple local chatbot for the provided financial policy PDF.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1 # on Windows
pip install -r requirements.txt
```

## Build index (once)

```bash
python -m src.build_index --pdf "data/For Task - Policy file.pdf"
```

## Run the Streamlit app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Features

- Local PDF ingestion and chunking with page + section metadata.
- Vector search via FAISS + Sentence-Transformers (fallback: TF-IDF).
- Conversation memory: short/pronoun follow-ups are linked to prior topic.

---

Built 2025-08-21
