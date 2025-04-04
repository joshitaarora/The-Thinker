import argparse

from src.data.data_loader import get_data

import dotenv

dotenv.load_dotenv()


def validate_dataset(value):
    allowed = ["codesearchnet", "codesc"]
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"Invalid dataset '{value}'. Choose from {allowed}.")
    return value


def validate_limit(value):
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Limit must be a non-negative integer.")
    return ivalue


def main():
    parser = argparse.ArgumentParser(description="Process dataset input for a script.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=validate_dataset,
        help="Dataset to use: either 'codeserachnet' or 'codesc'",
    )
    parser.add_argument(
        "--limit",
        type=validate_limit,
        default=None,
        help="Optional limit on number of records (non-negative integer)",
    )

    args = parser.parse_args()
    print(f"Using dataset: {args.dataset}")
    if args.limit is not None:
        print(f"Limiting records to: {args.limit}")
    else:
        print("No limit specified")

    dataset = args.dataset
    limit = args.limit
    
    data = get_data(dataset, limit)



if __name__ == "__main__":
    main()
