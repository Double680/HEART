import sys
sys.path.append('./')

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import argparse

from modules.process_tables import TableStructure

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--cover', action='store_true')
parser.add_argument('--name', default='none', type=str)
parser.add_argument('--path', type=str, default='models/Qwen3-1.7B')
args = parser.parse_args()

model_path = args.path
if args.dev:
    dataset_type = "dev"
else:
    dataset_type = "test"
    
data_root = f"./datasets/multihiertt"
src_file = f"{data_root}/{dataset_type}.json"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

with open(src_file, "r") as file:
    dataset = json.loads(file.read())

def make_conversation(sample, tid):
    with open("augment/ext_template.txt", "r") as file:
        template = file.read()

    content = template.replace("<QUESTION>", sample["qa"]["question"])

    for i in range(len(sample["paragraphs"])):
        if sample["paragraphs"][i] == f"## Table {tid} ##":
            table_tree = TableStructure(sample["tables"][tid], tid, sample["table_description"])
            table_instruction = sample["paragraphs"][i-1]
            content = content.replace("<INSTRUCTION>", table_instruction)

            table_content = sample["tables"][tid]
            content = content.replace("<TABLE>", table_content)

            table_row_headers = "Row Headers (ID: Name): \n" + table_tree.list_row_headers() + "\n"
            table_col_headers = "Column Headers (ID: Name): \n" + table_tree.list_col_headers() + "\n"
            table_headers = table_row_headers + table_col_headers
            content = content.replace("<HEADERS>", table_headers)
            
            break

    messages = [{
        "role": "user", 
        "content": content
    }]

    return messages

def ensure_dirs(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

for item in tqdm(dataset):
    ensure_dirs(f"./stored/{item["uid"]}")
    save_file_path = f"./stored/{item["uid"]}/table_ext_{args.name}.jsonl"
    if not args.cover and os.path.exists(save_file_path):
        continue
    with open(save_file_path, "w") as file:
        for tid in range(len(item["tables"])):
            messages = make_conversation(item, tid)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
            generated_ids = model.generate(
                **model_inputs,
                temperature=0.7,
                top_p=0.95,
                max_new_tokens=256
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            subtables = content.split("<answer>")[-1].split("</answer>")[0].strip('\n')
            rows = subtables.split("<rows>")[-1].split("</rows>")[0].strip().split(' | ')
            rids = []
            for row in rows:
                try:
                    rids.append(int(row))
                except Exception:
                    continue

            columns = subtables.split("<columns>")[-1].split("</columns>")[0].strip().split(' | ')
            cids = []
            for column in columns:
                try:
                    cids.append(int(column))
                except Exception:
                    continue

            subtable_item = {"tid": tid, "rids": rids, "cids": cids}
            
            file.write(json.dumps(subtable_item))
            file.write('\n')
