import os
import json
import torch
import torch.nn as nn
from modules.process_tables import *


class GroundTruthRetriever:
    def __init__(self, args):
        self.tabform = args.tabform

    def get_tabform_table_gth(self, sample):
        tables = sample['tables']
        table_description = sample["table_description"]
        table_trees = process_table_trees(tables, table_description)
        
        table_keys = sample['qa']['table_evidence']
        table_gth_ids = {
            i: {"rows": [], "cols": []}
            for i in range(len(table_trees))
        }
        for item in table_keys:
            ids = item.split('-')
            tid = int(ids[0])
            rid = int(ids[1])
            cid = int(ids[2])
            table_gth_ids[tid]["rows"].append(rid)
            table_gth_ids[tid]["cols"].append(cid)

        subtables = []
        for tid, table_tree in enumerate(table_trees):
            gth_rows = table_gth_ids[tid]["rows"]
            gth_cols = table_gth_ids[tid]["cols"]
            if len(gth_rows) == 0 and len(gth_cols) == 0:
                subtables.append('NONE')
                continue
            
            extend_gth_rows = []
            for row in gth_rows:
                row_id = row
                while True:
                    extend_gth_rows.append(row_id)
                    if table_tree.row_parents[row_id] == row_id:
                        break
                    row_id = table_tree.row_parents[row_id]
                    
            extract_rows = list(range(table_tree.row_header_boundary)) + extend_gth_rows
            extract_rows = list(set(extract_rows))
            extract_cols = list(range(table_tree.col_header_boundary)) + gth_cols
            extract_cols = list(set(extract_cols))
            subtable = table_tree.extract_subtable(extract_rows, extract_cols)
            subtables.append(subtable)
            
        return subtables


    def retrieve(self, sample):
        paragraphs = sample["paragraphs"]
        text_table_inds = []
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                if self.tabform:
                    text_table_inds.append(i-1)
                text_table_inds.append(i)
                table_cnt += 1
        text_inds = sample['qa']['text_evidence']
        update_text_inds = sorted(list(
            set(text_table_inds).union(set(text_inds))
        ))
        update_texts = [paragraphs[ind] for ind in update_text_inds]

        if self.tabform:
            update_tables = self.get_tabform_table_gth(sample)
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
        self.top_p = args.top_p
        self.tabform = args.tabform
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
        table_desc = sample['table_description']
        table_desc_st = [(int(key.split('-')[0]), table_desc[key]) for key in table_desc]
        table_scores = self.sim_func(question_emb, table_st_embs)
        if self.top_p == 0:
            retrieved_table_inds = torch.topk(table_scores, k=min(self.top_k, len(table_scores))).indices.tolist()
        else:
            retrieved_table_inds = torch.where(table_scores >= self.top_p)[0].tolist()
        update_table_inds = sorted(retrieved_table_inds)
        update_table_desc = [table_desc_st[ind] for ind in update_table_inds]
        update_table_dict = {i: [] for i in range(len(tables))}
        for table_id, desc in update_table_desc:
            update_table_dict[table_id].append(desc)
        update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]

        return update_tables, retrieved_table_inds

    def retrieve_tabform_table_evidence(self, sample, table_process_path):
        tables = sample['tables']
        table_description = sample["table_description"]
        table_trees = process_table_trees(tables, table_description)

        result_tables = []

        with open(table_process_path, "r") as file:
            process_txt = file.readlines()
            process_dic = {}
            for txt in process_txt:
                try:
                    txt_json = json.loads(txt)
                    process_dic[txt_json["tid"]] = {
                        "rids": txt_json["rids"],
                        "cids": txt_json["cids"]
                    }
                except Exception:
                    continue

        subtables = {}
        for i in range(len(tables)):
            if i in process_dic:
                row_ids = process_dic[i]["rids"]
                col_ids = process_dic[i]["cids"]

                row_ids, col_ids = table_trees[i].extend_header_boundary(row_ids, col_ids)

                if self.extend_header:
                    row_ids = table_trees[i].extend_row_headers(row_ids)
                try:
                    subtable = table_trees[i].extract_subtable(row_ids, col_ids)
                except Exception:
                    subtable = 'NONE'

                subtables[i] = {"rids": row_ids, "cids": col_ids}
            else:
                subtable = 'NONE'

            result_tables.append(subtable)

        return result_tables, subtables

    def get_question_emb(self, uid, query_type=None):
        if query_type is None:
            query_emb_path = os.path.join(self.path_root, uid, "query_embs.json")
        else:
            query_emb_path = os.path.join(self.path_root, uid, f"{query_type}_query_embs.json")
        with open(query_emb_path, "r") as file:
            query_emb_dict = json.loads(file.read())
            question_emb = torch.tensor(query_emb_dict["query_embs"]).to(self.device)
        return question_emb

    def retrieve(self, sample):
        uid = sample['uid']
        doc_emb_path = os.path.join(self.path_root, uid, "doc_embs.json")
        with open(doc_emb_path, "r") as file:
            emb_dict = json.loads(file.read())

        text_st_embs = torch.tensor(emb_dict["text_embs"]).to(self.device)
        table_st_embs = torch.tensor(emb_dict["table_embs"]).to(self.device)
        
        # text aug type
        if self.aug in ["raw_aug", "text-hard", "text-soft", "joint-hard", "joint-soft"]:
            text_question_emb = self.get_question_emb(uid, self.aug)
        elif self.aug in ["none", "table-hard", "table-soft"]:
            text_question_emb = self.get_question_emb(uid)
        elif self.aug == "text-hard-table-soft":
            text_question_emb = self.get_question_emb(uid, "text-hard")
        else:
            mode = self.aug.split('-')[-1]
            text_question_emb = self.get_question_emb(uid, f"text-{mode}")

        # table aug type
        if self.aug in ["raw_aug", "table-hard", "table-soft", "joint-hard", "joint-soft"]:
            table_question_emb = self.get_question_emb(uid, self.aug)
        elif self.aug in ["none", "text-hard", "text-soft"]:
            table_question_emb = self.get_question_emb(uid)
        elif self.aug == "text-hard-table-soft":
            table_question_emb = self.get_question_emb(uid, "table-soft")
        else:
            mode = self.aug.split('-')[-1]
            table_question_emb = self.get_question_emb(uid, f"table-{mode}")

        update_texts, retrieved_text_inds = self.retrieve_text_evidence(sample, text_question_emb, text_st_embs)
        update_tables, retrieved_table_inds = self.retrieve_table_evidence(sample, table_question_emb, table_st_embs)

        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds
        
         
        