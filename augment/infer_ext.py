import sys
sys.path.append('./')

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--name', type=str)
parser.add_argument('--path', type=str, default='models/Qwen3-1.7B')
parser.add_argument('--id', type=int)
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

def make_conversation(example):
    with open("augment/ext_template.txt", "r") as file:
        template = file.read()

    content = template.replace("<QUESTION>", example["qa"]["question"])
    tabular_content = ""
    tid = 0
    for i in range(len(example["paragraphs"])):
        if example["paragraphs"][i] == f"## Table {tid} ##":
            tabular_content += f"Table {tid} - "
            tabular_content += example["paragraphs"][i-1]
            tabular_content += example["tables"][tid]
            tid += 1
    content = content.replace("<TABLES>", tabular_content)

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
    messages = make_conversation(item)
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
    subtable_item = {
        "subtables": subtables
    }
    ensure_dirs(f"./stored/{item["uid"]}")
    with open(f"./stored/{item["uid"]}/table_{args.name}.json", "w") as file:
        file.write(json.dumps(subtable_item))
        file.write('\n')
    cnt += 1
