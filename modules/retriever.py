import os
import json
import torch
import torch.nn as nn
from modules.process_tables import *

class GroundTruthRetriever:
    def __init__(self, args):
        self.tabform = args.tabform
        self.tabextract = args.tabextract

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
            if self.tabextract:
                update_tables = self.get_tabform_table_gth(sample)
            else:
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
        self.tabextract_type = args.tabextract_type
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
        retrieved_table_inds = torch.topk(table_scores, k=min(self.top_k, len(table_scores))).indices.tolist()
        update_table_inds = sorted(retrieved_table_inds)
        update_table_desc = [table_desc_st[ind] for ind in update_table_inds]
        update_table_dict = {i: [] for i in range(len(tables))}
        for table_id, desc in update_table_desc:
            update_table_dict[table_id].append(desc)
        update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]

        return update_tables, retrieved_table_inds

    def retrieve_tabform_table_evidence(self, sample, tabextract_path):
        tables = sample['tables']
        table_description = sample["table_description"]
        table_trees = process_table_trees(tables, table_description)

        result_subtables = []
        if self.tabextract_type != 'none':
            with open(tabextract_path, "r") as file:
                try:
                    subtables = json.loads(file.read())["subtables"]
                    subtables = eval(subtables)
                except Exception:
                    subtables = {}
            for i in range(len(tables)):
                if i in subtables:
                    row_ids = subtables[i]["rid"]
                    col_ids = subtables[i]["cid"]
                    try:
                        subtable = table_trees[i].extract_subtable(row_ids, col_ids)
                    except Exception:
                        subtable = 'NONE'
                else:
                    subtable = 'NONE'
                result_subtables.append(subtable)
        else:
            pass
            # TBD
        return result_subtables, subtables
        

    def retrieve(self, sample):
        uid = sample['uid']
        doc_emb_path = os.path.join(self.path_root, uid, "doc_embs.json")
        with open(doc_emb_path, "r") as file:
            emb_dict = json.loads(file.read())
        text_st_embs = torch.tensor(emb_dict["text_embs"]).to(self.device)

        if self.aug != 'none':
            query_emb_path = os.path.join(self.path_root, uid, f"{self.aug}_query_embs.json")
        else:
            query_emb_path = os.path.join(self.path_root, uid, "query_embs.json")
        with open(query_emb_path, "r") as file:
            query_emb_dict = json.loads(file.read())
            question_emb = torch.tensor(query_emb_dict["query_embs"]).to(self.device)

        update_texts, retrieved_text_inds = self.retrieve_text_evidence(sample, question_emb, text_st_embs)
        
        if self.tabform:
            if self.tabextract:
                tabextract_path = os.path.join(self.path_root, uid, f"table_{self.tabextract_type}.json")
                update_tables, retrieved_table_inds = self.retrieve_tabform_table_evidence(sample, tabextract_path)
            else:
                update_tables = sample['tables']
                retrieved_table_inds = None
        else:
            table_st_embs = torch.tensor(emb_dict["table_embs"]).to(self.device)
            update_tables, retrieved_table_inds = self.retrieve_table_evidence(sample, question_emb, table_st_embs)

        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds
        
         
        