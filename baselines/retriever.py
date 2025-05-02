import os
import json
import torch
import torch.nn as nn
from openai import AsyncOpenAI

class DensePassageRetriever:
    def __init__(self, path_root, top_p=0.6, gpu=0):
        self.path_root = path_root
        self.top_p = top_p
        self.device = f"cuda:{gpu}" if gpu != -1 else "cpu"
        self.sim_func = nn.CosineSimilarity(dim=-1)

    def retrieve(self, sample):
        uid = sample['uid']
        emb_path = os.path.join(self.path_root, f"{uid}.json")
        with open(emb_path, "r") as file:
            emb_dict = json.loads(file.read())
        text_st_embs = torch.tensor(emb_dict["text_st_embs"]).to(self.device)
        table_st_embs = torch.tensor(emb_dict["table_st_embs"]).to(self.device)
        question_emb = torch.tensor(emb_dict["question_emb"]).to(self.device)
        
        # retrieve
        paragraphs = sample["paragraphs"]
        text_table_inds = []
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                text_table_inds.append(i)
                table_cnt += 1
        
        text_scores = self.sim_func(question_emb, text_st_embs)
        retrieved_text_inds = torch.where(text_scores > self.top_p)[0].tolist()
        update_text_inds = sorted(list(set(text_table_inds).union(set(retrieved_text_inds))))
        update_texts = [sample["paragraphs"][ind] for ind in update_text_inds]
        
        tables, table_desc = sample['tables'], sample['table_description']
        table_desc_st = [(int(key.split('-')[0]), table_desc[key]) for key in table_desc]

        table_scores = self.sim_func(question_emb, table_st_embs)
        table_inds = torch.where(table_scores > self.top_p)[0].tolist()
        update_table_desc = [table_desc_st[ind] for ind in table_inds]
        update_table_dict = {i: [] for i in range(len(tables))}
        for table_id, desc in update_table_desc:
            update_table_dict[table_id].append(desc.split(f"Table {table_id} shows ")[-1])
        update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]    

        return update_texts, update_tables

        # table_scores = self.sim_func(question_emb, table_st_embs)
        # table_inds = torch.where(table_scores > self.top_p)[0]
         
        