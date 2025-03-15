# Data Download Instructions

The dataset required for The-Thinker is too large to be stored in the repository. Follow these steps to download and organize the data.

## Data Structure
```
- src/data/
    - raw_data_code_search_net/  # Placeholder for dataset 1
    - raw_data_codes/  # Placeholder for dataset 2
    - unifier/  # Scripts to standardize datasets
```

## Download Instructions
1. **Dataset 1: CodeSearchNet**
   - Download from: [https://github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet)
   - Extract to `src/data/raw_data_code_search_net/`

2. **Dataset 2: Custom Code Repository Dataset**
   - Download from: (Specify your source, e.g., internal storage, cloud bucket, etc.)
   - Extract to `src/data/raw_data_codes/`

## Data Standardization
- The `unifier/` module helps convert datasets into a common format.
- Ensure that raw data follows the same structure before running `prepare.py`.

## Notes
- **Data is not included in Git**: Ensure datasets are stored locally.
- **Check storage space**: Some datasets may require significant disk space.
- **For issues**: Open an issue in GitHub Discussions or contact the maintainer.

