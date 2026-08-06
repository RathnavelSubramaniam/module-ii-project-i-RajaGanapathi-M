# AI-Powered Healthcare Decision Support System (RAG-based)

A Retrieval-Augmented Generation (RAG) system built to help healthcare professionals get fast, accurate, evidence-based answers from medical reference material — using the **Merck Manuals** (4,000+ pages) as the trusted knowledge base.

## 📌 Problem Statement

### Business Context
Healthcare professionals face growing information overload while needing quick, reliable access to medical knowledge for accurate diagnosis and treatment decisions — especially in time-sensitive situations. Reliable, up-to-date, and traceable information from trusted sources (like the Merck Manuals) is essential for maintaining high standards of care.

### Objective
Develop a RAG-based AI solution using the Merck Manuals to:
- **Understand** the problem of information overload in healthcare
- **Apply** AI/NLP techniques to streamline decision-making
- **Analyze** the impact on diagnostic support and patient outcomes
- **Evaluate** the solution's potential to standardize care
- **Create** a working prototype demonstrating feasibility

### Sample Questions the System Answers
1. What is the protocol for managing sepsis in a critical care unit?
2. What are the common symptoms of appendicitis, and how is it treated?
3. What are effective treatments for sudden patchy hair loss (alopecia)?
4. What treatments are recommended for traumatic brain injury?

## 🗂 Data
- **Source:** Merck Manuals (medical reference published by Merck & Co., available as a single PDF, 4,000+ pages / 23 sections)
- Covers disorders, diagnoses, tests, and drug information across specialties

## ⚙️ Approach

The notebook compares two approaches to answering medical questions:

**1. LLM with Prompt Engineering (Baseline)**
- Uses LLaMA-2 13B Chat (GGUF, via `llama-cpp-python`) with a system prompt and few-shot example
- Relies purely on the model's pre-trained knowledge — no external context

**2. Retrieval-Augmented Generation (RAG)**
- Loads and parses the Merck Manuals PDF (`PyMuPDFLoader`)
- Chunks the document (`RecursiveCharacterTextSplitter`, 1000 chars, 150 overlap)
- Embeds chunks using `all-MiniLM-L6-v2` (Sentence Transformers)
- Stores embeddings in a **ChromaDB** vector store
- Retrieves top-k relevant chunks per query and grounds the LLM's answer in that retrieved context

**3. Evaluation**
- An LLM-as-judge framework scores each response on:
  - **Groundedness** — is the answer supported by the retrieved context?
  - **Relevance** — does the answer fully address the question?
- Scores are parsed and compared across both approaches in a summary DataFrame

## 🛠 Tech Stack
- **LLM:** LLaMA-2 13B Chat (GGUF) via `llama-cpp-python`
- **Framework:** LangChain (`langchain`, `langchain_community`, `langchain_openai`)
- **Vector Store:** ChromaDB
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **PDF Parsing:** PyMuPDF (`pymupdf`)
- **Data Handling:** pandas, numpy
- **Environment:** Google Colab (GPU recommended)

## 📊 Key Results
- RAG-based responses consistently showed **higher groundedness** (typically 4–5) than prompt-engineered responses (typically 1), since answers were directly backed by retrieved manual content.
- RAG-based responses also achieved **higher relevance scores**, more completely addressing the clinical questions asked.
- Prompt-only responses, while covering general clinical knowledge, often lacked specific, verifiable citations from the source material.

## 💡 Business Recommendations
- Adopt RAG-based systems for medical information retrieval to improve factual grounding and traceability of AI-generated answers.
- Expand the knowledge base beyond a single manual to cover more specialties and updated guidelines.
- Explore improved chunking strategies and alternative embedding models to further boost retrieval quality.
- Pilot the system in a clinical decision-support setting with human-in-the-loop review before wider deployment.

