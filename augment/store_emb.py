import argparse
import torch
import json
import os
from augment.retriever import Retriever

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--doc', action='store_true')
parser.add_argument('--tabheader', action='store_true')
parser.add_argument('--query', action='store_true')
parser.add_argument('--raw_aug', action='store_true')
parser.add_argument('--grpo_aug', action='store_true')
args = parser.parse_args()

retriever_model_path = 'models/qwen3-embedding-0.6b'
dataset_type = "dev" if args.dev else "test"
ret = Retriever(retriever_model_path)

with open(f"datasets/multihiertt/{dataset_type}.json", "r") as file:
    data = json.loads(file.read())

if args.doc:
    for item in data:
        uid = item['uid']
        save_root = f'stored/{uid}'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        text_embs = ret.get_emb(item['paragraphs']).tolist()
        table_docs = [item['table_description'][key] for key in item['table_description']]
        table_embs = ret.get_emb(table_docs).tolist()
        doc_sample = {
            "text_embs": text_embs,
            "table_embs": table_embs
        }
        with open(f"{save_root}/doc_embs.json", "w") as file:
            file.write(json.dumps(doc_sample))

if args.query:
    for item in data:
        uid = item['uid']
        save_root = f'stored/{uid}'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        query_embs = ret.get_emb(item['qa']['question'], query_type=True).tolist()
        query_sample = {
            "query_embs": query_embs
        }
        with open(f"{save_root}/query_embs.json", "w") as file:
            file.write(json.dumps(query_sample))

if args.raw_aug:
    with open(f"datasets/multihiertt/{dataset_type}_raw_aug.jsonl", "r") as file:
        query_data = [json.loads(item) for item in file.readlines()]

    for item in query_data:
        uid = item['uid']
        save_root = f'store/{uid}'
        query_embs = ret.get_emb(item['new_query'], query_type=True).tolist()
        sample = {
            "query_embs": query_embs
        }
        with open(f"{save_root}/raw_aug_query_embs.json", "w") as file:
            file.write(json.dumps(sample))

if args.grpo_aug:
    with open(f"datasets/multihiertt/{dataset_type}_grpo_aug.jsonl", "r") as file:
        query_data = [json.loads(item) for item in file.readlines()]

    for item in query_data:
        uid = item['uid']
        save_root = f'store/{uid}'
        query_embs = ret.get_emb(item['new_query'], query_type=True).tolist()
        sample = {
            "query_embs": query_embs
        }
        with open(f"{save_root}/grpo_aug_query_embs.json", "w") as file:
            file.write(json.dumps(sample))