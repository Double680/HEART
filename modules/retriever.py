import os
import json
import torch
import torch.nn as nn
import math
import re
from collections import Counter
from modules.process_tables import *


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_scores(scores):
    if len(scores) == 0:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def top_k_indices(scores, top_k):
    if len(scores) == 0:
        return []
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in indexed_scores[:min(top_k, len(indexed_scores))]]


def threshold_or_top_k_indices(scores, top_k, top_p):
    if len(scores) == 0:
        return []
    if top_p > 0:
        return [idx for idx, score in enumerate(scores) if score >= top_p]
    return top_k_indices(scores, top_k)


def tensor_to_scores(scores):
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu().reshape(-1).tolist()
    return list(scores)


class BM25Scorer:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = [tokenize(doc) for doc in documents]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in self.documents) / len(self.documents) if self.documents else 0
        self.doc_freq = Counter()
        for doc in self.documents:
            self.doc_freq.update(set(doc))

    def score(self, query):
        query_terms = tokenize(query)
        doc_count = len(self.documents)
        if doc_count == 0:
            return []
        scores = []
        for doc in self.documents:
            term_freq = Counter(doc)
            doc_len = len(doc)
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * freq * (self.k1 + 1) / denom
            scores.append(score)
        return scores


class EvidenceRanker:
    def score_documents(self, query, documents):
        raise NotImplementedError

    def retrieve(self, query, documents, top_k=10):
        return top_k_indices(self.score_documents(query, documents), top_k)

    def soft_retrieve_eval(self, query, documents, gth):
        scores = normalize_scores(self.score_documents(query, documents))
        return self.score_relevance(scores, gth)

    def score_relevance(self, scores, gth):
        if not scores:
            return 1.0 if not gth else 0.0

        gth = set(gth)
        positive_score = 1.0
        for idx, score in enumerate(scores):
            if idx in gth:
                positive_score += score
        positive_score /= len(gth) + 1
        return positive_score

    def eval(self, indices, gth):
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().reshape(-1).tolist()
        indices = list(dict.fromkeys(indices).keys())
        gth = list(dict.fromkeys(gth).keys())
        joint = set(indices).intersection(gth)
        precision = len(joint) / len(indices) if indices else 0.0
        recall = len(joint) / len(gth) if gth else 0.0
        if len(gth) == 0:
            ndcg = 1.0
        else:
            dcg, idcg = 0.0, 0.0
            for i, idx in enumerate(indices):
                if idx in gth:
                    dcg += 1 / math.log2(i + 2)
            for i in range(min(len(gth), len(indices))):
                idcg += 1 / math.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
        return precision, recall, ndcg


class DenseEvidenceRanker(EvidenceRanker):
    def __init__(self, model_name, device="cuda", torch_dtype="auto"):
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        model_kwargs = {"torch_dtype": dtype}
        if device.startswith("cuda"):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"}
        )
        self.model.eval()

    def get_emb(self, inputs, query_type=False):
        if query_type:
            return self.model.encode([inputs], prompt_name="query")
        return self.model.encode(inputs)

    def score_documents(self, query, documents):
        if not documents:
            return []
        query_embedding = self.model.encode([query], prompt_name="query")
        document_embeddings = self.model.encode(documents)
        scores = self.model.similarity(query_embedding, document_embeddings).squeeze(0)
        return tensor_to_scores(scores)

    def soft_retrieve_eval(self, query, documents, gth):
        return self.score_relevance(self.score_documents(query, documents), gth)


class BM25EvidenceRanker(EvidenceRanker):
    def score_documents(self, query, documents):
        return BM25Scorer(documents).score(query)


class HybridEvidenceRanker(EvidenceRanker):
    def __init__(self, model_name, device="cuda", bm25_weight=0.5, torch_dtype="auto"):
        self.dense_ranker = DenseEvidenceRanker(model_name, device=device, torch_dtype=torch_dtype)
        self.bm25_weight = bm25_weight

    def get_emb(self, inputs, query_type=False):
        return self.dense_ranker.get_emb(inputs, query_type=query_type)

    def score_documents(self, query, documents):
        dense_scores = normalize_scores(self.dense_ranker.score_documents(query, documents))
        bm25_scores = normalize_scores(BM25Scorer(documents).score(query))
        return [
            self.bm25_weight * bm25_score + (1 - self.bm25_weight) * dense_score
            for dense_score, bm25_score in zip(dense_scores, bm25_scores)
        ]


def build_evidence_ranker(retrieve_type, model_name=None, device="cuda", bm25_weight=0.5):
    if retrieve_type in ["dense", "dpr"]:
        return DenseEvidenceRanker(model_name, device=device)
    if retrieve_type == "bm25":
        return BM25EvidenceRanker()
    if retrieve_type == "hybrid":
        return HybridEvidenceRanker(model_name, device=device, bm25_weight=bm25_weight)
    raise ValueError(f"Unsupported evidence ranker type: {retrieve_type}")


def table_description_items(sample):
    table_desc = sample['table_description']
    return [
        (int(key.split('-')[0]), int(key.split('-')[1]), int(key.split('-')[2]), table_desc[key])
        for key in table_desc
    ]


def build_text_context(sample, retrieved_text_inds, tabform=False):
    paragraphs = sample["paragraphs"]
    text_table_inds = []
    table_cnt = 0
    for i in range(len(paragraphs)):
        if paragraphs[i] == f'## Table {table_cnt} ##':
            if tabform and i > 0:
                text_table_inds.append(i-1)
            text_table_inds.append(i)
            table_cnt += 1
    update_text_inds = sorted(list(
        set(text_table_inds).union(set(retrieved_text_inds))
    ))
    update_texts = [paragraphs[ind] for ind in update_text_inds]
    return update_texts


def build_table_context(sample, retrieved_table_inds, tabform=False):
    tables = sample['tables']
    table_desc_st = table_description_items(sample)
    update_table_inds = sorted(retrieved_table_inds)
    update_table_desc = [table_desc_st[ind] for ind in update_table_inds]
    if tabform:
        update_tables = []
        tabform_dict = {}
        for tid, rid, cid, _ in update_table_desc:
            if tid not in tabform_dict:
                tabform_dict[tid] = {"rids": [], "cids": []}
            tabform_dict[tid]["rids"].append(rid)
            tabform_dict[tid]["cids"].append(cid)
        table_trees = process_table_trees(tables, sample['table_description'])
        for tid, table_tree in enumerate(table_trees):
            if tid in tabform_dict:
                row_ids = list(set(tabform_dict[tid]["rids"]))
                col_ids = list(set(tabform_dict[tid]["cids"]))
                row_ids, col_ids = table_tree.extend_header_boundary(row_ids, col_ids)
                row_ids = table_tree.extend_row_headers(row_ids)
                subtable = table_tree.extract_subtable(row_ids, col_ids)
            else:
                subtable = ''
            update_tables.append(subtable)
        return update_tables, tabform_dict

    update_table_dict = {i: [] for i in range(len(tables))}
    for tid, _, _, desc in update_table_desc:
        update_table_dict[tid].append(desc)
    update_tables = ["\n".join(update_table_dict[i]) for i in range(len(tables))]
    return update_tables, retrieved_table_inds


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
        self.device = args.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

    def retrieve_text_evidence(self, sample, question_emb, text_st_embs):
        text_scores = self.sim_func(question_emb, text_st_embs)
        if len(text_scores) == 0:
            return [], []
        retrieved_text_inds = torch.topk(text_scores, k=min(self.top_k, len(text_scores))).indices.tolist()
        update_texts = build_text_context(sample, retrieved_text_inds, self.tabform)

        return update_texts, retrieved_text_inds

    def retrieve_table_evidence(self, sample, question_emb, table_st_embs):
        table_scores = self.sim_func(question_emb, table_st_embs)
        if len(table_scores) == 0:
            empty_tables = ['' for _ in range(len(sample['tables']))]
            return empty_tables, {} if self.tabform else []
        if self.top_p == 0:
            retrieved_table_inds = torch.topk(table_scores, k=min(self.top_k, len(table_scores))).indices.tolist()
        else:
            retrieved_table_inds = torch.where(table_scores >= self.top_p)[0].tolist()
        update_tables, retrieved_table_inds = build_table_context(sample, retrieved_table_inds, self.tabform)
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

    def get_text_question_emb(self, uid):
        if self.aug in ["raw_aug", "text-hard", "text-soft", "joint-hard", "joint-soft", "hybrid", "joint-soft-0.1", "joint-soft-cont", "joint-soft-0.1-cont"]:
            return self.get_question_emb(uid, self.aug)
        if self.aug in ["none", "table-hard", "table-soft"]:
            return self.get_question_emb(uid)
        if self.aug == "text-hard-table-soft":
            return self.get_question_emb(uid, "text-hard")
        mode = self.aug.split('-')[-1]
        return self.get_question_emb(uid, f"text-{mode}")

    def get_table_question_emb(self, uid):
        if self.aug in ["raw_aug", "table-hard", "table-soft", "joint-hard", "joint-soft", "hybrid", "joint-soft-0.1", "joint-soft-cont", "joint-soft-0.1-cont"]:
            return self.get_question_emb(uid, self.aug)
        if self.aug in ["none", "text-hard", "text-soft"]:
            return self.get_question_emb(uid)
        if self.aug == "text-hard-table-soft":
            return self.get_question_emb(uid, "table-soft")
        mode = self.aug.split('-')[-1]
        return self.get_question_emb(uid, f"table-{mode}")

    def retrieve(self, sample):
        uid = sample['uid']
        doc_emb_path = os.path.join(self.path_root, uid, "doc_embs.json")
        with open(doc_emb_path, "r") as file:
            emb_dict = json.loads(file.read())

        text_st_embs = torch.tensor(emb_dict["text_embs"]).to(self.device)
        table_st_embs = torch.tensor(emb_dict["table_embs"]).to(self.device)
        
        text_question_emb = self.get_text_question_emb(uid)
        table_question_emb = self.get_table_question_emb(uid)

        update_texts, retrieved_text_inds = self.retrieve_text_evidence(sample, text_question_emb, text_st_embs)
        update_tables, retrieved_table_inds = self.retrieve_table_evidence(sample, table_question_emb, table_st_embs)

        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds


class BM25Retriever:
    def __init__(self, args):
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tabform = args.tabform

    def get_query(self, sample):
        return sample.get("augmented_question", sample['qa']['question'])

    def retrieve_text_evidence(self, sample, query):
        scores = BM25Scorer(sample["paragraphs"]).score(query)
        retrieved_text_inds = top_k_indices(scores, self.top_k)
        update_texts = build_text_context(sample, retrieved_text_inds, self.tabform)
        return update_texts, retrieved_text_inds

    def retrieve_table_evidence(self, sample, query):
        table_docs = [item[-1] for item in table_description_items(sample)]
        scores = normalize_scores(BM25Scorer(table_docs).score(query))
        retrieved_table_inds = threshold_or_top_k_indices(scores, self.top_k, self.top_p)
        update_tables, retrieved_table_inds = build_table_context(sample, retrieved_table_inds, self.tabform)
        return update_tables, retrieved_table_inds

    def retrieve(self, sample):
        query = self.get_query(sample)
        update_texts, retrieved_text_inds = self.retrieve_text_evidence(sample, query)
        update_tables, retrieved_table_inds = self.retrieve_table_evidence(sample, query)
        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds


class HybridRetriever(DensePassageRetriever):
    def __init__(self, args):
        super().__init__(args)
        self.bm25_weight = args.bm25_weight

    def get_query(self, sample):
        return sample.get("augmented_question", sample['qa']['question'])

    def combine_scores(self, dense_scores, bm25_scores):
        dense_scores = normalize_scores(dense_scores)
        bm25_scores = normalize_scores(bm25_scores)
        return [
            self.bm25_weight * bm25_score + (1 - self.bm25_weight) * dense_score
            for dense_score, bm25_score in zip(dense_scores, bm25_scores)
        ]

    def retrieve_text_evidence(self, sample, query, question_emb, text_st_embs):
        dense_scores = self.sim_func(question_emb, text_st_embs).tolist()
        bm25_scores = BM25Scorer(sample["paragraphs"]).score(query)
        scores = self.combine_scores(dense_scores, bm25_scores)
        retrieved_text_inds = top_k_indices(scores, self.top_k)
        update_texts = build_text_context(sample, retrieved_text_inds, self.tabform)
        return update_texts, retrieved_text_inds

    def retrieve_table_evidence(self, sample, query, question_emb, table_st_embs):
        table_docs = [item[-1] for item in table_description_items(sample)]
        dense_scores = self.sim_func(question_emb, table_st_embs).tolist()
        bm25_scores = BM25Scorer(table_docs).score(query)
        scores = self.combine_scores(dense_scores, bm25_scores)
        retrieved_table_inds = threshold_or_top_k_indices(scores, self.top_k, self.top_p)
        update_tables, retrieved_table_inds = build_table_context(sample, retrieved_table_inds, self.tabform)
        return update_tables, retrieved_table_inds

    def retrieve(self, sample):
        uid = sample['uid']
        doc_emb_path = os.path.join(self.path_root, uid, "doc_embs.json")
        with open(doc_emb_path, "r") as file:
            emb_dict = json.loads(file.read())

        text_st_embs = torch.tensor(emb_dict["text_embs"]).to(self.device)
        table_st_embs = torch.tensor(emb_dict["table_embs"]).to(self.device)
        text_question_emb = self.get_text_question_emb(uid)
        table_question_emb = self.get_table_question_emb(uid)
        query = self.get_query(sample)

        update_texts, retrieved_text_inds = self.retrieve_text_evidence(sample, query, text_question_emb, text_st_embs)
        update_tables, retrieved_table_inds = self.retrieve_table_evidence(sample, query, table_question_emb, table_st_embs)
        return update_texts, update_tables, retrieved_text_inds, retrieved_table_inds
