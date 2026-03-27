<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=🧠%20docqnatool&fontSize=60&fontAlign=50" alt="docqnatool">
</p>

<p align="center">
  <a href="https://github.com/crimsonKn1ght/docqnatool/stargazers">
    <img src="https://img.shields.io/github/stars/crimsonKn1ght/docqnatool?style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/crimsonKn1ght/docqnatool/network/members">
    <img src="https://img.shields.io/github/forks/crimsonKn1ght/docqnatool?style=for-the-badge" alt="GitHub forks">
  </a>
  <a href="https://github.com/crimsonKn1ght/docqnatool/graphs/commit-activity">
    <img src="https://img.shields.io/maintenance/yes/2025?style=for-the-badge" alt="Maintained">
  </a>
  <a href="https://github.com/crimsonKn1ght/docqnatool">
    <img src="https://img.shields.io/github/languages/top/crimsonKn1ght/docqnatool?style=for-the-badge" alt="Language">
  </a>
</p>

---

## Overview

**docqnatool** is a document Q&A assistant built with Streamlit. Upload PDF, DOCX, or TXT files, and ask questions about their content. The tool extracts and indexes text from your documents (including OCR for scanned images), then uses a Groq-hosted LLaMA model to answer your questions with relevant context.

Try the hosted version: [docqnatool.streamlit.app](https://docqnatool.streamlit.app/)

---

## How it works

1. Uploaded documents are parsed and text is extracted (with optional OCR via Tesseract).
2. Text is split into chunks and embedded using a TF-IDF vectorizer.
3. Chunks are stored in a FAISS vector index for fast similarity search.
4. When you ask a question, the top matching chunks are retrieved and passed as context to `llama-3.3-70b-versatile` via the Groq API.
5. The model returns an answer grounded in your document content.

---

## Features

- Supports PDF, DOCX, and TXT file formats
- OCR support via Tesseract for scanned images embedded in documents
- Duplicate file detection using MD5 hashing
- Chat-style interface with message history
- Per-session file stats (file count, word count, total size)

---

## Requirements

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- A [Groq API key](https://console.groq.com/)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/crimsonKn1ght/docqnatool.git
cd docqnatool
```

**2. Install system dependencies**

On Debian/Ubuntu:
```bash
sudo apt-get install $(cat packages.txt)
```

On macOS (Homebrew):
```bash
brew install tesseract tesseract-lang
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**

```bash
export GROQ_API_KEY=your_key_here
```

Or create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

---

## Usage

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

1. Use the sidebar to upload one or more documents (PDF, DOCX, or TXT).
2. Toggle OCR on or off depending on whether your documents contain scanned images.
3. Once processing is complete, type your question in the chat box.
4. Use "Clear All Documents" in the sidebar to reset the session.

---

## Configuration

The following values can be changed at the top of `app.py` under the `CONSTANTS` section:

| Constant | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model to use for answering |
| `LLM_TEMPERATURE` | `0` | Model temperature (0 = deterministic) |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `SIMILARITY_TOP_K` | `5` | Number of chunks retrieved per question |

---

## Dev container

A `.devcontainer` config is included for use with VS Code or GitHub Codespaces. It targets Python 3.11, installs all dependencies automatically, and exposes port 8501.

---

## License

MIT. See [LICENSE](LICENSE) for details.
