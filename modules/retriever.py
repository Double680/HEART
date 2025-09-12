import os
import json
import torch
import torch.nn as nn
from utils.table_header import *
from utils.subtable_generator import *

class GroundTruthRetriever:
    def __init__(self, args):
        self.tabform = args.tabform

    def retrieve(self, sample):
        paragraphs = sample["paragraphs"]
        text_table_inds = []
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                text_table_inds.append(i)
                table_cnt += 1
        text_inds = sample['qa']['text_evidence']
        update_text_inds = sorted(list(
            set(text_table_inds).union(set(text_inds))
        ))
        update_texts = [paragraphs[ind] for ind in update_text_inds]

        if self.tabform:
            update_tables = sample['tables']
            table_inds = None
        else:
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
                update_table_dict[table_id].append(desc)
            update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]    
        
        return update_texts, update_tables, text_inds, table_inds


class DensePassageRetriever:
    def __init__(self, args):
        self.path_root = args.stored_embs_path
        self.aug = args.query_aug
        self.top_k = args.top_k
        self.tabform = args.tabform
        self.tabextract = args.tabextract
        self.sim_func = nn.CosineSimilarity(dim=-1)
        self.device = "cuda"       

    def retrieve_text_evidence(self, sample, question_emb, text_st_embs):
        paragraphs = sample["paragraphs"]
        text_table_inds = []
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                if self.tabform:
                    text_table_inds.append(i-1)
                text_table_inds.append(i)
                table_cnt += 1
        
        text_scores = self.sim_func(question_emb, text_st_embs)
        retrieved_text_inds = torch.topk(text_scores, k=min(self.top_k, len(text_scores))).indices.tolist()
        update_text_inds = sorted(list(
            set(text_table_inds).union(set(retrieved_text_inds))
        )) 
        update_texts = [sample["paragraphs"][ind] for ind in update_text_inds]

        return update_texts, retrieved_text_inds

    def retrieve_table_evidence(self, sample, question_emb, table_st_embs):
        tables = sample['tables']
        # if self.tabheader:
        #     table_doc = get_table_docs(sample['table_headers'])
        #     col_sites, row_sites = get_site_lists(table_doc)
        #     col_sim_scores = self.sim_func(question_emb, table_col_embs)
        #     col_indices = torch.topk(col_sim_scores, k=min(self.top_k, col_sim_scores.size(-1))).indices.squeeze(0)
        #     col_indices = [col_sites[ind] for ind in col_indices.tolist()]
        #     row_sim_scores = self.sim_func(question_emb, table_row_embs)
        #     row_indices = torch.topk(row_sim_scores, k=min(self.top_k, row_sim_scores.size(-1))).indices.squeeze(0)
        #     row_indices = [row_sites[ind] for ind in row_indices.tolist()]
        #     retrieved_table_inds = [col_indices, row_indices]
        #     update_tables = []
        #     col_indices_dict = {i: [] for i in range(len(tables))}
        #     for i, _, col_site in col_indices:
        #         col_indices_dict[i].extend(col_site)
        #     row_indices_dict = {i: [] for i in range(len(tables))}
        #     for i, _, row_site in row_indices:
        #         row_indices_dict[i].extend(row_site)
        #     for i in range(len(tables)):
        #         table_html = tables[i]
        #         if i not in sample['table_headers_max_ids']:
        #             update_tables.append('None')
        #             continue
        #         row_header_length = sample['table_headers_max_ids'][i]['row']
        #         row_site_base = [i for i in range(row_header_length)]
        #         row_site_retreival = row_indices_dict[i]
        #         row_sites_final = sorted(list(set(row_site_base + row_site_retreival)))
        #         col_header_length = sample['table_headers_max_ids'][i]['col']
        #         col_site_base = [i for i in range(col_header_length)]   
        #         col_site_retreival = col_indices_dict[i]
        #         col_sites_final = sorted(list(set(col_site_base + col_site_retreival)))
        #         table_html = extract_subtable(table_html, row_sites_final, col_sites_final)
        #         update_tables.append(table_html)
        # else:
        table_desc = sample['table_description']
        table_desc_st = [(int(key.split('-')[0]), table_desc[key]) for key in table_desc]
        table_scores = self.sim_func(question_emb, table_st_embs)
        retrieved_table_inds = torch.topk(table_scores, k=min(self.top_k, len(table_scores))).indices.tolist()
        update_table_inds = sorted(retrieved_table_inds)
        update_table_desc = [table_desc_st[ind] for ind in update_table_inds]
        update_table_dict = {i: [] for i in range(len(tables))}
        for table_id, desc in update_table_desc:
            update_table_dict[table_id].append(desc)
        update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]

        return update_tables, retrieved_table_inds

    def retrieve_tabform_table_evidence(self, sample, question_emb):
        pass

    def retrieve(self, sample):
        uid = sample['uid']
        doc_emb_path = os.path.join(self.path_root, uid, "doc_embs.json")
        with open(doc_emb_path, "r") as file:
            emb_dict = json.loads(file.read())
        text_st_embs = torch.tensor(emb_dict["text_embs"]).to(self.device)
        # if self.tabheader:
        #     table_header_emb_path = os.path.join(self.path_root, uid, "table_header_embs.json")
        #     with open(table_header_emb_path, "r") as file:
        #         header_emb_dict = json.loads(file.read())
        #     table_col_embs = torch.tensor(header_emb_dict["table_col_embs"]).to(self.device)
        #     table_row_embs = torch.tensor(header_emb_dict["table_row_embs"]).to(self.device)
        # else:
        table_st_embs = torch.tensor(emb_dict["table_embs"]).to(self.device)
        if self.aug != 'none':
            query_emb_path = os.path.join(self.path_root, uid, f"{self.aug}_query_embs.json")
        else:
            query_emb_path = os.path.join(self.path_root, uid, "query_embs.json")
        with open(query_emb_path, "r") as file:
            query_emb_dict = json.loads(file.read())
            question_emb = torch.tensor(query_emb_dict["query_embs"]).to(self.device)
        
        # retrieve
        update_texts, retrieved_text_inds = self.retrieve_text_evidence(sample, question_emb, text_st_embs)
        if self.tabform:
            if self.tabextract:
                update_tables, retrieved_table_inds = self.retrieve_tabform_table_evidence(sample, question_emb)
            else:
                update_tables = sample['tables']
                retrieved_table_inds = None
        else:
            update_tables, retrieved_table_inds = self.retrieve_table_evidence(sample, question_emb, table_st_embs)

        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds
        
         
        