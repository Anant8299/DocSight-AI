# 🔍 DocSight AI

### Multimodal RAG for Visual Document Understanding

DocSight AI is an interactive **multimodal Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF documents and ask questions about their content.

Instead of relying only on extracted text, DocSight AI processes PDF pages as images, uses **ColPali** for visual document retrieval, stores embeddings in **ChromaDB**, and uses **Google Gemini** to generate answers from the retrieved document pages.

---

## 🚀 Features

- 📄 Upload multiple PDF documents directly through the web interface
- 🔎 Visual document retrieval using ColPali
- 🗄️ Persistent ChromaDB vector storage
- 🤖 Gemini multimodal question answering
- 💬 Conversational chat history
- 📚 Search across all uploaded documents or selected documents
- 🖼️ Display retrieved source pages with answers
- ♻️ Duplicate document detection using content hashing
- 🗑️ Delete individual documents or clear the complete document library
- 📊 Live indexing progress for uploaded documents
- 🌐 Interactive Streamlit interface

---

## 🧠 How It Works

```text
                    PDF Upload
                        │
                        ▼
                PDF → Page Images
                        │
                        ▼
              ColPali Page Embeddings
                        │
                        ▼
                    ChromaDB
                        │
                        │
                 User Question
                        │
                        ▼
              ColPali Query Embedding
                        │
                        ▼
                Similarity Search
                        │
                        ▼
               Relevant PDF Pages
                        │
                        ▼
             Gemini Multimodal Model
                        │
                        ▼
              Answer + Source Pages
```

### Pipeline

1. **PDF Upload** — Users can upload multiple PDF documents through the Streamlit interface.
2. **Page Processing** — Each PDF is converted into page images.
3. **Visual Embedding** — ColPali generates visual embeddings for each document page.
4. **Vector Storage** — Page embeddings and document metadata are stored in ChromaDB.
5. **Query Processing** — The user's question is converted into a ColPali query embedding.
6. **Retrieval** — ChromaDB retrieves the most relevant pages using cosine similarity.
7. **Multimodal Generation** — Retrieved page images are provided to Gemini along with the user's question.
8. **Response** — The application generates an answer and displays the retrieved source pages.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web interface |
| PyTorch | Deep learning inference |
| ColPali | Visual document and query embeddings |
| ChromaDB | Vector storage and similarity retrieval |
| Google Gemini | Multimodal answer generation |
| pdf2image | PDF-to-image conversion |
| Pillow | Image processing |

---

## 📂 Project Structure

```text
DocSight-AI/
│
├── app.py
└── README.md
```

`app.py` contains the complete Streamlit application, including PDF ingestion, visual embedding, retrieval, document management, and multimodal question answering.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anant8299/DocSight-AI.git
cd DocSight-AI
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The project requires Poppler for PDF-to-image conversion.

For Ubuntu/Debian:

```bash
sudo apt-get install -y poppler-utils
```

---

## 🔑 Gemini API Key

The application requires a Google Gemini API key.

You can enter the API key through the Streamlit sidebar when the application starts.

Alternatively, set it as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key"
```

**Never commit your API key to GitHub.**

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open at the local Streamlit URL, usually:

```text
http://localhost:8501
```

---

## 💡 Usage

1. Launch the Streamlit application.
2. Upload one or more PDF documents.
3. Wait for the indexing process to complete.
4. Select specific documents if required.
5. Ask a question in the chat interface.
6. View the generated answer along with the retrieved source pages.
7. Continue asking follow-up questions using the conversational interface.

---

## 📌 Example Use Cases

DocSight AI can be used for question answering over visually rich documents such as:

- 📑 Research papers
- 📊 Reports containing charts and tables
- 📚 Academic notes
- 📖 Technical documentation
- 🧾 Business documents
- 📋 Manuals and presentations

---

## ⚠️ Current Limitations

The current retrieval implementation uses **mean-pooled 128-dimensional ColPali embeddings with cosine similarity** rather than the full late-interaction MaxSim retrieval mechanism of the original ColPali approach.

The application therefore represents a practical engineering implementation of visual document RAG rather than a complete reproduction of the original ColPali retrieval architecture.

---

## 👨‍💻 Author

**Anant Pandey**

Multimodal Document Understanding & RAG Project
