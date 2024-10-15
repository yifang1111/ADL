import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--test_file", type=str, default='test.json', help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--mc_ckpt", type=str, default='mc_roberta2', help="A file containing the multiple choice checkpoint."
    )
    parser.add_argument(
        "--output_file", type=str, default='new_test_roberta2.json', help="A new csv or a json file containing the test data."
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Load the provided test.json and result.json
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(os.path.join(args.mc_ckpt,'results.json'), 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    # Create a new list to store updated test data
    updated_test_data = []

    # Iterate through the test data and update with the relevant paragraphs based on result.json
    for i, item in enumerate(test_data):
        # Append the relevant paragraph based on the result data
        relevant_paragraph = item["paragraphs"][result_data[i]]
        updated_item = {
            **item,  # Include all original keys
            "relevant": relevant_paragraph  # Add the 'relevant' key with the appropriate value
        }
        updated_test_data.append(updated_item)

    # Save the updated test.json with the relevant paragraphs
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_test_data, f, ensure_ascii=False, indent=4)

    print("Save new test file")

if __name__ == "__main__":
    main()