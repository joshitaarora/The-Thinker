from datasets import load_dataset
import os
import json

# Define languages and splits
languages = ['python', 'java', 'javascript', 'go', 'ruby', 'php']
splits = {
    'train': 'train',
    'test': 'test',
    'validation': 'valid'  # Map validation to valid
}

# Load the dataset
dataset = load_dataset('code_search_net')

# Create directories and export data for each language and split
for lang in languages:
    for split_src, split_dest in splits.items():
        # Create directory
        directory = f"src/data/raw_data_code_search_net/resources/data/{lang}/final/jsonl/{split_dest}"
        os.makedirs(directory, exist_ok=True)
        
        # Filter by language
        lang_data = dataset[split_src].filter(lambda x: x['language'] == lang)
        
        # Get number of examples
        num_examples = len(lang_data)
        print(f"Processing {num_examples} examples for {lang}/{split_dest}")
        
        # Define output file path
        output_path = f"{directory}/{lang}_{split_dest}.jsonl"
        
        # Export to jsonl format
        with open(output_path, 'w') as f:
            for item in lang_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved to {output_path}")

print("All data exported successfully!")