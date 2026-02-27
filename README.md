# MTTQA

# Enhancing Information Retrieval for Table-Centered Question Answering with Text in Technical Documents

This repository contains the code and resources for the M.Sc. thesis project by **Ghazaleh Keivani Heshajani** at the University of Isfahan, Shahreza Campus, under the supervision of Dr. Maryam Lotfi and Dr. Maryam Hosseini.

## 📌 Project Overview

Technical documents in industrial domains (e.g., steel, oil & gas, electronics) often contain a mix of **semi‑structured tables** and **unstructured text**. Current knowledge management systems and search engines fail to effectively retrieve and reason over such heterogeneous data, leading to costly reliance on human memory and delayed operational decisions.

This research aims to **enhance information retrieval for question answering over hybrid technical documents** by combining:

- **Agentic AI** – multi‑agent systems that plan, reason, and use external tools.
- **Retrieval‑Augmented Generation (RAG)** – integrating vector and graph‑based retrieval.
- **Knowledge Graphs** – capturing implicit relationships between tables and text.
- **Multimodal understanding** – processing tables, text, and potentially images/layout.

We focus on the **TAT‑QA dataset** – a real‑world financial QA benchmark requiring numerical reasoning over tables and associated paragraphs. Our approach will be evaluated on similar industrial‑style documents.

## 🎯 Objectives

1. **Extract and align** complex tables and surrounding text from noisy technical documents using layout‑aware models (e.g., DocLLM, LayoutLM).
2. **Design a multi‑agent architecture** (inspired by MAPLE, MACT, G‑MACT) where specialised agents handle planning, retrieval, computation, and verification.
3. **Implement hybrid RAG** that combines dense vector retrieval with knowledge graph traversal to retrieve relevant table cells and text spans.
4. **Answer complex multi‑step questions** requiring numerical reasoning (addition, subtraction, comparison, aggregation) over combined evidence.
5. **Evaluate** on TAT‑QA and possibly a custom industrial dataset using a comprehensive set of metrics.

## 🧠 Methodology

The proposed system is built on **LangGraph** for agent orchestration and **LangChain** for tool integration. The main workflow consists of:

- **Planning Agent**: decomposes the question into a sequence of sub‑tasks (e.g., filter rows, retrieve text, compute ratio).
- **Table Parser Agent**: extracts structured information from tables (handles multi‑level headers, merged cells).
- **Text Retriever Agent**: performs dense retrieval over the associated paragraphs (using a per‑sample vector store).
- **Computation Agent**: writes and executes Python code for numerical operations.
- **Verification Agent**: checks the consistency of the answer with the evidence and produces the final answer.
- **Memory**: a short‑term memory (via LangGraph state) stores intermediate results and reasoning steps.

We experiment with both **single‑agent** (ReAct) and **multi‑agent** (MAPLE‑like) architectures, and compare **vector‑only RAG** with **graph‑enhanced RAG**.

## 📊 Dataset

Primary dataset: **[TAT‑QA](https://github.com/NExTplusplus/TAT-QA)**  
- 16,552 QA pairs over 2,757 tables from real financial reports.
- Each sample contains a table (with row/col headers and numbers) and ≥2 descriptive paragraphs.
- Questions require numerical reasoning (addition, subtraction, multiplication, division, counting, comparison) and often combine table and text.
- Answer types: span, multi‑span, arithmetic expression, counting.

We also plan to explore scientific tables (SciTab) or custom industrial documents.

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangChain** – tool definitions, RAG pipelines
- **LangGraph** – multi‑agent state graphs
- **HuggingFace Transformers** – for layout‑aware models (e.g., LayoutLMv3, DocLLM)
- **FAISS / Chroma** – vector stores
- **NetworkX / Neo4j** – knowledge graph storage and querying
- **Pandas / NumPy** – data manipulation
- **OpenAI / LLaMA / Qwen** – LLM backends (open‑weight models preferred)
- **Evaluation libraries**: ROUGE, BLEU, sentence‑transformers, scikit‑learn

## 📁 Repository Structure

. ├── agents/ # Agent definitions (planner, parser, retriever, etc.) ├── data/ # Dataset loaders and preprocessors (TAT‑QA) ├── graphs/ # LangGraph workflow definitions ├── retrieval/ # Dense and graph‑based retrieval modules ├── evaluation/ # Metrics (EM, F1, ROUGE, BLEU, precision@k, recall@k, MAP) ├── notebooks/ # Exploratory analysis and demos (including evaluation notebooks) ├── configs/ # Configuration files (LLM endpoints, embedding models) ├── requirements.txt └── README.md


## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/table-text-qa-agent.git
   cd table-text-qa-agent


   2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the TAT‑QA dataset**:
   - Visit [https://nextplusplus.github.io/TAT-QA/](https://nextplusplus.github.io/TAT-QA/) and download the JSON files (`train.json`, `dev.json`, `test.json`).
   - Place them in `data/tatqa/`.

4. **Set up environment variables**:
   Create a `.env` file with your OpenAI API key (if using OpenAI models) or configure your local model paths.

5. **Run a baseline experiment**:
   ```bash
   python run_experiment.py --config configs/baseline.yaml
   ```

6. **Run evaluation on a subset of the dev set**:
   Open the notebook `notebooks/evaluation_demo.ipynb` and execute the cells to see metrics for a few samples.

## 📈 Evaluation Metrics

We report a comprehensive set of metrics covering both **retrieval** and **generation** aspects:

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Binary: 1 if the normalized answer exactly matches the ground truth. |
| **F1 Score** | Harmonic mean of precision and recall at token level. |
| **ROUGE‑1 / ROUGE‑2 / ROUGE‑L** | Overlap of unigrams, bigrams, and longest common subsequence. |
| **BLEU** | Precision‑based n‑gram overlap (smoothed). |
| **Cosine Similarity** | Semantic similarity between sentence embeddings of prediction and ground truth. |
| **Precision@k** | Fraction of retrieved top‑k paragraphs that are relevant. |
| **Recall@k** | Fraction of relevant paragraphs retrieved in the top‑k. |
| **Mean Average Precision (MAP)** | Average precision across all relevant paragraphs. |

These metrics are computed per question and then averaged. The evaluation script (`evaluation/evaluate.py`) and the accompanying notebook provide detailed results.

## 📊 Preliminary Results

On a small subset of the TAT‑QA dev set (5 samples, 1 question each), we obtained the following averages:

| Metric          | Value |
|-----------------|-------|
| **EM**          | 0.20  |
| **F1**          | 0.20  |
| **ROUGE‑1 F1**  | 0.20  |
| **BLEU**        | 0.04  |
| **Cosine Similarity** | 0.47 |
| **Precision@3** | 0.33  |
| **Recall@3**    | 0.81  |
| **MAP**         | 2.24* |

*MAP currently exceeds 1 due to duplicate retrieved paragraph IDs (e.g., `[4,4,5]`). Deduplication is required to obtain a valid MAP score (expected ≤1).

**Observations:**
- The retriever consistently finds one relevant paragraph in the top‑3 (precision@3 = 0.33) and, for most samples, all relevant paragraphs (recall@3 = 1.0).
- Generation metrics (EM, F1, ROUGE, BLEU) are low for most samples, often zero, despite moderate semantic similarity (cosine similarity). This indicates issues in token‑level matching, likely due to list‑to‑string conversion, punctuation differences, or stopword removal.
- Sample 0 (gross margin) achieved perfect EM, F1, ROUGE‑1, and cosine similarity, demonstrating the model's ability to extract exact numeric values when the answer is a simple number.

**Current limitations:**
- Gold paragraph IDs are heuristically defined by checking if the answer string appears in a paragraph. This may be inaccurate and produce false positives/negatives.
- Duplicate retrieved IDs artificially inflate MAP; deduplication is needed.
- Token‑level metrics for longer descriptive answers are sensitive to minor variations (e.g., `contract’s` vs `contract's`).

Future work will address these issues and expand evaluation to more samples.

## 📄 Related Work

- **TAT‑QA**: Zhu et al. (2021) – dataset.
- **MAPLE**: Bai et al. (2025) – multi‑agent adaptive planning with memory.
- **MACT / G‑MACT**: Zhou et al. (2025) – efficient multi‑agent collaboration with tool use.
- **ReAcTable**: Zhang et al. (2024) – enhancing ReAct for table QA.
- **DocLLM**: Wang et al. (2023) – layout‑aware document LLM.

## 👥 Supervision

- **Dr. Maryam Lotfi** (Primary Supervisor) – Shahreza Higher Education Center
- **Dr. Maryam Hosseini** (Co‑Supervisor) – Shahreza Higher Education Center

## 📝 License

This project is for academic research purposes. The TAT‑QA dataset is available for non‑commercial use under its own license.

## 📬 Contact

For questions or collaboration, please contact:  
Ghazaleh Keivani Heshajani – [email address]  
University of Isfahan, Shahreza Campus

---

**Note:** This repository is under active development for a master's thesis. Results and code will be updated as the research progresses.
```
