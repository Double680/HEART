import sys
sys.path.append('./')

import argparse
import json
import os
from augment.retriever import Retriever
from augment.table_tree_docs import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--doc', action='store_true')
parser.add_argument('--tabform', action='store_true')
parser.add_argument('--query_aug', type=str)
parser.add_argument('--tree_aug', action='store_true')
args = parser.parse_args()

retriever_model_path = 'models/Qwen3-Embedding-0.6B'
dataset_type = "dev" if args.dev else "test"
ret = Retriever(retriever_model_path)

with open(f"datasets/multihiertt/{dataset_type}.json", "r") as file:
    data = json.loads(file.read())

if args.doc:
    for item in tqdm(data):
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

if args.tabform:
    with open(f"datasets/multihiertt/{dataset_type}_headers.json", "r") as file:
        header_dict = json.loads(file.read())
    for item in tqdm(data):
        uid = item['uid']
        save_root = f'stored/{uid}'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        table_header = header_dict[uid]
        table_docs = get_table_tree_docs(table_header)
        col_doc_texts, row_doc_texts = [], []
        for table in table_docs:
            col_doc_texts += [table['col_docs'][id]['text'] for id in range(len(table['col_docs']))]
            row_doc_texts += [table['row_docs'][id]['text'] for id in range(len(table['row_docs']))]
        col_embs = ret.get_emb(col_doc_texts).tolist()
        row_embs = ret.get_emb(row_doc_texts).tolist()
        header_embs = {
            "table_col_embs": col_embs,
            "table_row_embs": row_embs
        }
        with open(f"{save_root}/table_header_embs.json", "w") as file:
            file.write(json.dumps(header_embs))


# if args.query:
#     for item in tqdm(data):
#         uid = item['uid']
#         save_root = f'stored/{uid}'
#         if not os.path.exists(save_root):
#             os.mkdir(save_root)
#         query_embs = ret.get_emb(item['qa']['question'], query_type=True).tolist()
#         query_sample = {
#             "query_embs": query_embs
#         }
#         with open(f"{save_root}/query_embs.json", "w") as file:
#             file.write(json.dumps(query_sample))

with open(f"datasets/multihiertt/{dataset_type}_{args.query_aug}.jsonl", "r") as file:
    query_data = [json.loads(item) for item in file.readlines()]

for item in tqdm(query_data):
    uid = item['uid']
    save_root = f'stored/{uid}'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    query_embs = ret.get_emb(item['new_query'], query_type=True).tolist()
    sample = {
        "query_embs": query_embs
    }
    with open(f"{save_root}/{args.query_aug}_query_embs.json", "w") as file:
        file.write(json.dumps(sample))

# if args.grpo_aug:
#     with open(f"datasets/multihiertt/{dataset_type}_grpo_aug.jsonl", "r") as file:
#         query_data = [json.loads(item) for item in file.readlines()]

#     for item in tqdm(query_data):
#         uid = item['uid']
#         save_root = f'stored/{uid}'
#         if not os.path.exists(save_root):
#             os.mkdir(save_root)
#         query_embs = ret.get_emb(item['new_query'], query_type=True).tolist()
#         sample = {
#             "query_embs": query_embs
#         }
#         with open(f"{save_root}/grpo_aug_query_embs.json", "w") as file:
#             file.write(json.dumps(sample))

if args.tree_aug:
    with open(f"datasets/multihiertt/{dataset_type}_tree_aug.jsonl", "r") as file:
        query_data = [json.loads(item) for item in file.readlines()]

    for item in tqdm(query_data):
        uid = item['uid']
        save_root = f'stored/{uid}'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        query_embs = ret.get_emb(item['new_query'], query_type=True).tolist()
        sample = {
            "query_embs": query_embs
        }
        with open(f"{save_root}/tree_aug_query_embs.json", "w") as file:
            file.write(json.dumps(sample))