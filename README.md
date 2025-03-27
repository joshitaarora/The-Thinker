# The-Thinker

The-Thinker is a machine learning research project focused on leveraging LLMs for code understanding, summarization, and embedding-based retrieval.

## Features
- **Chunking Code Repositories**: Processes repositories into structured chunks.
- **Summarization with LLMs**: Generates summaries of code snippets.
- **Embeddings & Vector DB**: Stores and queries code representations.
- **Data Standardization**: Unifies datasets into a common format.

## Project Structure
```
- src
    - data
        - README.md  # How to download datasets (data is not included in commits)
        - raw_data_code_search_net/  # Placeholder for dataset 1
        - raw_data_codes/  # Placeholder for dataset 2
        - unifier/  # Standardizes datasets (TODO: Define format)
    - code
        - chunker.py  # Splits repositories into chunks
        - chunk_summaries.py  # Generates LLM-based summaries
        - embedder.py  # Creates embeddings for chunked code
        - vector_db.py  # Stores and retrieves embeddings
- prepare.py  # Prepares data (setup, preprocessing)
- run.py  # Runs the end-to-end pipeline
- test.py  # Unit tests for modules
- generate_test_cases.py  # Creates test cases for validation
- generate_documentation.py  # Automates documentation generation
```

## Installation
```sh
pip install -r requirements.txt
```

Create .env from .env.sample and fill in the necessary values.


## Usage
1. **Prepare the data** (Follow [Data README](src/data/README.md) for downloading datasets)
   ```sh
   python prepare.py
   ```
2. **Run the full pipeline**
   ```sh
   python run.py
   ```
3. **Run tests**
   ```sh
   python test.py
   ```

## Data Handling
Large datasets are not included in the repository. Follow the instructions in [`src/data/README.md`](https://github.com/joshitaarora/The-Thinker/blob/main/src/data/README.md) to download the necessary files.

## Contribution Guidelines
- **Branching**: Always work on a separate branch (never push directly to `master`).
- **Pull Requests**: All changes must go through PRs and be reviewed.
- **Code Style**: Follow PEP8 and maintain modularity.
- **Issues**: Report bugs and suggest improvements via GitHub Issues.

## Roadmap
- Define a standardized data format for unification.
- Improve chunking strategies for better summarization.
- Optimize vector search efficiency.

---
**Repository:** [GitHub - The-Thinker](https://github.com/joshitaarora/The-Thinker)

**Maintainer:** Daniyal

For any queries, open an issue or reach out via GitHub Discussions.

