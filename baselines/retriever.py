import os
import json
import torch
import torch.nn as nn


class GroundTruthRetriever:
    def __init__(self):
        pass

    def retrieve(self, sample):
        paragraphs = sample["paragraphs"]
        text_table_inds = []
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                text_table_inds.append(i)
                table_cnt += 1
        text_inds = sample['qa']['text_evidence']
        update_text_inds = sorted(list(set(text_table_inds).union(set(text_inds))))
        update_texts = [paragraphs[ind] for ind in update_text_inds]

        table_keys = sample['qa']['table_evidence']
        tables = [sample['table_description'][item] for item in table_keys]
        table_inds = []
        for i, key in enumerate(sample['table_description']):
            if key in table_keys:
                table_inds.append(i)
        tables, table_desc = sample['tables'], sample['table_description']
        table_desc_st = [(int(key.split('-')[0]), table_desc[key]) for key in table_desc]
        update_table_desc = [table_desc_st[ind] for ind in table_inds]
        update_table_dict = {i: [] for i in range(len(tables))}
        for table_id, desc in update_table_desc:
            update_table_dict[table_id].append(desc.split(f"Table {table_id} shows ")[-1])
        update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]    
        
        return update_texts, update_tables, text_inds, table_inds


class DensePassageRetriever:
    def __init__(self, path_root, aug='none', top_k=10, gpu=0):
        self.path_root = path_root
        self.aug = aug
        self.top_k = top_k
        self.device = f"cuda:{gpu}" if gpu != -1 else "cpu"
        self.sim_func = nn.CosineSimilarity(dim=-1)

    def retrieve(self, sample):
        uid = sample['uid']
        doc_emb_path = os.path.join(self.path_root, uid, "doc_embs.json")
        with open(doc_emb_path, "r") as file:
            emb_dict = json.loads(file.read())
        text_st_embs = torch.tensor(emb_dict["text_embs"]).to(self.device)
        table_st_embs = torch.tensor(emb_dict["table_embs"]).to(self.device)
        if self.aug != 'none':
            query_emb_path = os.path.join(self.path_root, uid, f"{self.aug}_query_embs.json")
        else:
            query_emb_path = os.path.join(self.path_root, uid, "query_embs.json")
        with open(query_emb_path, "r") as file:
            query_emb_dict = json.loads(file.read())
            question_emb = torch.tensor(query_emb_dict["query_embs"]).to(self.device)
        
        # retrieve
        paragraphs = sample["paragraphs"]
        text_table_inds = []
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                text_table_inds.append(i)
                table_cnt += 1
        
        text_scores = self.sim_func(question_emb, text_st_embs)
        try:
            retrieved_text_inds = sorted(torch.topk(text_scores, k=self.top_k).indices.tolist()) # torch.where(text_scores > self.top_p)[0].tolist()
        except Exception:
            retrieved_text_inds = [i for i in range(len(text_scores))]
        update_text_inds = sorted(list(set(text_table_inds).union(set(retrieved_text_inds))))
        update_texts = [sample["paragraphs"][ind] for ind in update_text_inds]
        
        tables, table_desc = sample['tables'], sample['table_description']
        table_desc_st = [(int(key.split('-')[0]), table_desc[key]) for key in table_desc]

        table_scores = self.sim_func(question_emb, table_st_embs)
        try:
            retrieved_table_inds = sorted(torch.topk(table_scores, k=self.top_k).indices.tolist())
        except Exception:
            retrieved_table_inds = [i for i in range(len(table_scores))]
        update_table_desc = [table_desc_st[ind] for ind in retrieved_table_inds]
        update_table_dict = {i: [] for i in range(len(tables))}
        for table_id, desc in update_table_desc:
            update_table_dict[table_id].append(desc.split(f"Table {table_id} shows ")[-1])
        update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]    

        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds

        # table_scores = self.sim_func(question_emb, table_st_embs)
        # table_inds = torch.where(table_scores > self.top_p)[0]
         
        