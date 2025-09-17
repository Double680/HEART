import sys
sys.path.append('./')

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--name', type=str)
parser.add_argument('--path', type=str, default='models/Qwen3-1.7B')
args = parser.parse_args()

model_path = args.path
if args.dev:
    dataset_type = "dev"
else:
    dataset_type = "test"
    
data_root = f"./datasets/multihiertt"
src_file = f"{data_root}/{dataset_type}.json"
tgt_file = f"{data_root}/{dataset_type}_{args.name}.jsonl"

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
    template = "Modify the given question precisely by adding more details within <query> </query> tags. \nQuestion: <QUESTION> "
    messages = [{
        "role": "user", 
        "content": template.replace("<QUESTION>", example["qa"]["question"])
    }]
    return messages

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
    new_query = content.split("<query>")[-1].split("</query>")[0]
    new_query_item = {
        "uid": item["uid"],
        "new_query": new_query
    }
    with open(tgt_file, "a") as file:
        file.write(json.dumps(new_query_item))
        file.write('\n')
