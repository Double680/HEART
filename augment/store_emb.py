import sys
sys.path.append('./')

import argparse
import json
import os
from modules.retriever import DenseEvidenceRanker
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--doc', action='store_true')
parser.add_argument('--query', action='store_true')
parser.add_argument('--query_aug', type=str, default=None)
parser.add_argument('--data_root', type=str, default='datasets/multihiertt')
parser.add_argument('--stored_root', type=str, default='stored')
parser.add_argument('--model_path', type=str, default='models/Qwen3-Embedding-0.6B')
args = parser.parse_args()

dataset_type = "dev" if args.dev else "test"
ret = DenseEvidenceRanker(args.model_path)

with open(f"{args.data_root}/{dataset_type}.json", "r") as file:
    data = json.loads(file.read())

if args.doc:
    for item in tqdm(data):
        uid = item['uid']
        save_root = os.path.join(args.stored_root, uid)
        os.makedirs(save_root, exist_ok=True)
        text_embs = ret.get_emb(item['paragraphs']).tolist()
        table_docs = [item['table_description'][key] for key in item['table_description']]
        table_embs = ret.get_emb(table_docs).tolist()
        doc_sample = {
            "text_embs": text_embs,
            "table_embs": table_embs
        }
        with open(os.path.join(save_root, "doc_embs.json"), "w") as file:
            file.write(json.dumps(doc_sample))

if args.query:
    for item in tqdm(data):
        uid = item['uid']
        save_root = os.path.join(args.stored_root, uid)
        os.makedirs(save_root, exist_ok=True)
        query_embs = ret.get_emb(item['qa']['question'], query_type=True).tolist()
        query_sample = {
            "query_embs": query_embs
        }
        with open(os.path.join(save_root, "query_embs.json"), "w") as file:
            file.write(json.dumps(query_sample))

if args.query_aug is not None:
    with open(f"{args.data_root}/{dataset_type}_{args.query_aug}.jsonl", "r") as file:
        query_data = [json.loads(item) for item in file.readlines()]

    for item in tqdm(query_data):
        uid = item['uid']
        save_root = os.path.join(args.stored_root, uid)
        os.makedirs(save_root, exist_ok=True)
        query_embs = ret.get_emb(item['new_query'], query_type=True).tolist()
        sample = {
            "query_embs": query_embs
        }
        with open(os.path.join(save_root, f"{args.query_aug}_query_embs.json"), "w") as file:
            file.write(json.dumps(sample))
