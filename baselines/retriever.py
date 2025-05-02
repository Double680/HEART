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
            emb_dict = json.loads(file.read(emb_path))
        text_st_embs = torch.tensor(emb_dict["text_st_embs"]).to(self.device)
        table_st_embs = torch.tensor(emb_dict["table_st_embs"]).to(self.device)
        question_emb = torch.tensor(emb_dict["question_emb"]).to(self.device)
        
        # retrieve
        text_scores = self.sim_func(question_emb, text_st_embs)
        text_inds = torch.where(text_scores > self.top_p)[0]
        retrieved_text = [sample["paragraphs"][ind] for ind in text_inds.tolist()]
        sample["paragraphs"] = retrieved_text

        table_scores = self.sim_func(question_emb, table_st_embs)
        table_inds = torch.where(table_scores > self.top_p)[0]
         
        