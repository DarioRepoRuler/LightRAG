import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import re

WORKING_DIR = "./tower_of_hanoi"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2mm",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

with open("./tower_of_hanoi_dataset.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# # Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )
# Perform hybrid search
print("Response:")
response = rag.query("In tower of Hanoi puzzle with 3 disks.\
The Tower of Hanoi rules dictate that you must move one disk at a time, only the top disk, between three pegs while never placing a larger disk on a smaller one. \
Disk numbers represent disk sizes, where a larger number = a larger disk.\
Current state (from bottom to top):\
Source peg (S): [3, 2, 1]\
Auxiliary peg (A): []\
Target peg (T): []\
What is the next move to get all disks to the target peg T?\
Your answer must be formatted using the tag system: MDxYZ where:\
- M means Move\
- D followed by the disk number (e.g., Dx for disk x)\
- Source peg letter (S, A, or T)\
- Target peg letter (S, A, or T)\
For example, moving disk x from Source to Target would be MDxST.\
If no move is needed because all disks are already on the target peg, use NM for No Move.\
Answer with the tag system only.\
", param=QueryParam(mode="hybrid"))

print("Full Response:")
print(response)

print("\nExtracted Move:")

# Try first to find a move code that appears after "optimal" or "recommended"
final_move_pattern = r"(?:optimal|recommended).*?[{]?([MN][DM][0-9]+[AST]{2}|NM)[}]?"
match = re.search(final_move_pattern, response, re.IGNORECASE | re.DOTALL)

# If not found with the above pattern, try looking for markdown headings
if not match:
    heading_move_pattern = r"#{2,4}\s*[{]?([MN][DM][0-9]+[AST]{2}|NM)[}]?"
    match = re.search(heading_move_pattern, response)

# If still not found, get the last occurrence of any move code
if not match:
    all_matches = re.findall(r"[{]?([MN][DM][0-9]+[AST]{2}|NM)[}]?", response)
    if all_matches:
        move_code = all_matches[-1]
        print(move_code)
    else:
        print("No valid move code found in the response")
else:
    move_code = match.group(1)
    print(move_code)