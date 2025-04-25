# The Thinker

**The Thinker** is a retrieval-augmented generation (RAG) system designed for understanding and querying large-scale code repositories using abstract syntax trees (ASTs), smart chunking, and embedding-based retrieval with Milvus.

---

## 📝 Description

This paper presents **"The Thinker"**, a novel approach to building knowledge bases for code repositories using Abstract Syntax Trees (ASTs). Traditional code chunking methods treat code as linear text, fragmenting it arbitrarily and losing semantic relationships. Our AST-based approach preserves code structure and contextual boundaries, generating more meaningful embeddings for vector-based search powering semantic code search, which aligns with the goals of the CodeSearchNet Challenge (Husain et al., 2019).

Using the CodeSearchNet dataset, we demonstrate that AST-based chunking creates semantically coherent units that improve knowledge retrieval compared to traditional chunking methods. Through both manual evaluation and LLM-based assessment (Briggs, 2024), we show that our system consistently provides more relevant and accurate results for code-related queries.

Although the approach is computationally intensive, it offers significant advantages for documentation generation, semantic code search, and code-based chatbots. The Thinker architecture integrates with open-source tools and requires no proprietary dependencies, making it accessible for organizations seeking to improve code understanding and navigation within their repositories.

Our implementation and data processing scripts are available on GitHub.

---

## 📦 Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/joshitaarora/The-Thinker.git
cd The-Thinker
```

### 2. Create a virtual environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

```bash
cp .env.sample .env
# Edit .env to include your OpenAI API key or other necessary variables
```

---

## 🚀 Running the Project

### Step 1: Clone Dataset Repository

Download and prepare the [CodeSearchNet Challenge](https://github.com/github/CodeSearchNet) dataset:

```bash
python3 setup_data.py --dataset codesearchnet
```

This will download and clone all repositories with `depth == 1`.

---

### Step 2: Create Code Chunks

Smart or naive chunking of a specific repository:

```bash
python3 create_chunks.py --repo-path ./src/data/selected_repos/tensorflow --mode smart
```

- Replace the path with any repo directory you want.
- `--mode` can be `smart` or `naive`.

---

### Step 3: Generate Summaries, Embeddings, and Store in Milvus

Ensure Milvus is installed and running locally.

Then run:

```bash
# For naive chunking:
python3 create_embeddings_naive.py --dataset codesearchnet

# For smart chunking:
python3 create_embeddings.py --dataset codesearchnet
```

---

### Step 4: Query Using CLI Chatbot

Run the CLI-based chatbot for querying your indexed codebase:

```bash
python3 query.py --mode naive "Give me detailed logic of locate method?"
```

You can also use:

```bash
python3 query.py --mode smart "How is the backpropagation implemented?"
```

---

### Step 5: Run Analysis on Selected Repositories

For experiments and scoring comparisons:

```bash
python3 run_selected_experiment_scoring_combined.py
```

---

## 🤝 Contribution Guidelines

- **Branching**: Always work on a separate branch (never push directly to `master`).
- **Pull Requests**: All changes must go through PRs and be reviewed.
- **Code Style**: Follow PEP8 and maintain modularity.
- **Issues**: Report bugs and suggest improvements via GitHub Issues.

---

## 🗺️ Roadmap

- Define a standardized data format for unification.
- Improve chunking strategies for better summarization.
- Optimize vector search efficiency.

---

**Repository:** [GitHub - The-Thinker](https://github.com/joshitaarora/The-Thinker)
