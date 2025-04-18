import os
import json
import subprocess
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def clone_repo(github_url: str, destination_path: str) -> None:
    destination = Path(destination_path)
    if destination.exists():
        print(f"✅ Repository already exists at {destination}")
        return

    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", github_url, str(destination)],
            input=b"\n",  # Prevents hanging on prompts
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        if result.returncode != 0:
            stderr_output = result.stderr.decode("utf-8", errors="ignore")
            if "Username for" in stderr_output or "password" in stderr_output.lower():
                pass
                print(f"⏭️ Skipped private/auth-required repo: {github_url}")
            else:
                pass
                print(
                    f"❌ Clone failed for {github_url}: {stderr_output.strip().splitlines()[-1]}"
                )
        else:
            print(f"✅ Cloned: {github_url}")
            pass

    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout while cloning: {github_url}")
        pass
    except Exception as e:
        print(f"⚠️ Unexpected error while cloning {github_url}: {e}")
        pass


def count_total_entries(
    base_dir: str,
    languages: Optional[List[str]] = None,
    split_types: Optional[List[str]] = ["train", "test", "valid"],
) -> int:
    languages = languages or ["python", "javascript", "java", "ruby", "php", "go"]
    count = 0
    for lang in languages:
        lang_count = 0
        for split_type in split_types:
            split_count = 0
            print(f"Counting entries for language: {lang} and split: {split_type}")
            input_path = (
                Path(base_dir) / lang / "final" / "jsonl" / split_type / "unzipped"
            )
            for jsonl_file in input_path.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    split_count += sum(1 for _ in f)
            
            print(f"Count={split_count}")
            lang_count += split_count
        count += lang_count
    return count


def stream_codesearchnet_data(
    base_dir: str,
    languages: Optional[List[str]] = None,
    split_types: Optional[List[str]] = ["train", "test", "valid"],
):
    languages = languages or ["python", "javascript", "java", "ruby", "php", "go"]

    for lang in languages:
        for split_type in split_types:
            input_path = (
                Path(base_dir) / lang / "final" / "jsonl" / split_type / "unzipped"
            )
            for jsonl_file in input_path.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            parsed = json.loads(line)
                            parsed["github_repo"] = (
                                f"https://abbasidaniyal:test123@github.com/{parsed['repo']}.git"
                            )
                            yield parsed
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to parse line in {jsonl_file.name}")


def get_codesc_data(limit=None):
    # Placeholder for CodeSC dataset reader
    return []


def _clone_repo_entry(entry: dict, base_clone_path: str) -> dict:
    repo = entry["repo"]
    cloned_path = os.path.join(base_clone_path, repo)
    clone_repo(entry["github_repo"], cloned_path)
    entry["repo_path"] = cloned_path
    return entry


def get_data(
    dataset: str,
    limit: Optional[int] = None,
    clone_parallel: bool = True,
    languages: Optional[List[str]] = None,
    batch_size: int = 50,
    max_workers: int = 8,
):
    if dataset == "codesearchnet":
        base_dir = os.environ["CODE_SEARCH_NET_PATH"]
        unique_repos = set()
        data = []
        batch = []
        clone_base_path = "src/data/code_search_net_repos"
        Path(clone_base_path).mkdir(parents=True, exist_ok=True)

        # total_entries = count_total_entries(base_dir)
        stream = stream_codesearchnet_data(base_dir, languages)
        # pbar = tqdm(total=total_entries, desc="Processing unique repos")

        for i, entry in tqdm(enumerate(stream), total=450000):
            if i+1 % 10000 == 0:
                print(f"Processed {i+1} entries")

            repo = entry["repo"]
            if repo not in unique_repos:
                unique_repos.add(repo)
                batch.append(entry)
                # pbar.update(1)

                if len(batch) == batch_size:
                    if clone_parallel:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [
                                executor.submit(_clone_repo_entry, e, clone_base_path)
                                for e in batch
                            ]
                            for _ in as_completed(futures):
                                pass
                    else:
                        for e in batch:
                            _clone_repo_entry(e, clone_base_path)

                    data.extend(batch)
                    batch = []

            if limit and len(data) >= limit:
                break

        # Clone and store any remaining items in the last batch
        if batch:
            if clone_parallel:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_clone_repo_entry, e, clone_base_path)
                        for e in batch
                    ]
                    for _ in as_completed(futures):
                        pass
            else:
                for e in batch:
                    _clone_repo_entry(e, clone_base_path)
            data.extend(batch)

        # pbar.close()
        return data

    elif dataset == "codesc":
        return get_codesc_data(limit=limit)
    else:
        raise ValueError(f"Invalid dataset '{dataset}'")
