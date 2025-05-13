import json

def extract_text_to_txt(json_file_path, txt_file_path):
    """
    Extracts the 'question' and 'answer' text from a JSON file and writes it to a formatted TXT file.

    Args:
        json_file_path (str): The path to the input JSON file.
        txt_file_path (str): The path to the output TXT file.
    """

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return

    with open(txt_file_path, 'w') as outfile:
        for i, item in enumerate(data):
            question = item.get('question', 'N/A')  # Get question, default to 'N/A' if missing
            answer = item.get('answer', 'N/A')      # Get answer, default to 'N/A' if missing

            outfile.write(f"--- Entry {i + 1} ---\n")
            outfile.write(f"Question:\n{question}\n\n")
            outfile.write(f"Answer:\n{answer}\n")
            outfile.write("\n")  # Add extra spacing between entries

    print(f"Successfully extracted text to {txt_file_path}")


# Usage
json_file = 'tower_of_hanoi_dataset.json'  # Replace with your JSON file path
txt_file = 'tower_of_hanoi_dataset.txt'    # Replace with your desired TXT file path
extract_text_to_txt(json_file, txt_file)