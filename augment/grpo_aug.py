import sys
sys.path.append('./')

from accelerate import Accelerator
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from augment.retriever import Retriever
from augment.reranker_vllm import *
from modules.process_tables import *
from transformers import AutoModelForCausalLM
import torch
import argparse

def make_conversation(example):
    with open("augment/aug_template.txt", "r") as file:
        template = file.read()

    prompt = {
        "prompt": [{
            "role": "user", 
            "content": template.replace("<QUESTION>", example["qa"]["question"]) + '/no_think'
        }]
    }
    return prompt

def reward_value(value):
    return 4 * value
    
def format_reward(question):
    reward = -2
    if "<query>" in question and "</query>" in question:
        reward += 1
    if len(question.split("<query>")[-1].split("</query>")[0]) > 0:
        reward += 1
    return reward

def rerank_table_evidence(tables, table_description, question):
    table_trees = process_table_trees(tables, table_description)

    rerank_scores = {}
    for i in range(len(tables)):
        rerank_scores[i] = {"rids": {}, "cids": {}}
        table_tree = table_trees[i]

        row_floor, row_ceil = table_tree.row_header_boundary, table_tree.max_rows
        row_evids = []
        for rid in range(row_floor, row_ceil):
            row_evid = table_tree.extract_row(rid, extend=True)
            row_evids.append(row_evid)
        if len(row_evids):
            row_scores = call_reranker_vllm_online(question, row_evids)["results"]
        for rid in range(row_floor, row_ceil):
            rerank_scores[i]["rids"][rid] = row_scores[rid - row_floor]["relevance_score"]
 
        col_floor, col_ceil = table_tree.col_header_boundary, table_tree.max_cols
        col_evids = []
        for cid in range(col_floor, col_ceil):
            col_evid = table_tree.extract_col(cid)
            col_evids.append(col_evid)
        if len(col_evids):
            col_scores = call_reranker_vllm_online(question, col_evids)["results"]
        for cid in range(col_floor, col_ceil):
            rerank_scores[i]["cids"][cid] = col_scores[cid - col_floor]["relevance_score"]

    return rerank_scores

def get_rerank_overall_reward(pred_rerank_score, gth_evidence, alpha=2, beta=0.5):
    gth_rerank = {}
    gth_unhit_num = {"row": 0, "col": 0}
    gth_cnt = len(gth_evidence)
    for item in gth_evidence:
        tid, rid, cid = item.split('-')
        tid = int(tid); rid = int(rid); cid = int(cid)
        if tid not in gth_rerank:
            gth_rerank[tid] = {"rids": {}, "cids": {}}
            if rid not in gth_rerank[tid]["rids"]:
                gth_rerank[tid]["rids"][rid] = 0
                gth_unhit_num["row"] -= 1
            gth_rerank[tid]["rids"][rid] += 1
            if cid not in gth_rerank[tid]["cids"]:
                gth_rerank[tid]["cids"][cid] = 1
                gth_unhit_num["col"] -= 1
            gth_rerank[tid]["cids"][cid] += 1

    for tid in pred_rerank_score:
        tid_scores = pred_rerank_score[tid]
        gth_unhit_num["row"] += len(tid_scores["rids"])
        gth_unhit_num["col"] += len(tid_scores["cids"])

    item_reward = 0
    for tid in pred_rerank_score:
        tid_scores = pred_rerank_score[tid]
        for rid in tid_scores["rids"]:
            relevance = pred_rerank_score[tid]["rids"][rid] - reranker_lambda
            if tid in gth_rerank and rid in gth_rerank[tid]["rids"]:
                if relevance > 0:
                    item_reward += alpha * gth_rerank[tid]["rids"][rid] / gth_cnt if gth_cnt else 0
            else:
                if relevance < 0:
                    item_reward += beta / gth_unhit_num["row"] if gth_unhit_num["row"] else 0
        for cid in tid_scores["cids"]:
            relevance = pred_rerank_score[tid]["cids"][cid] - reranker_lambda
            if tid in gth_rerank and cid in gth_rerank[tid]["cids"]:
                if relevance > 0:
                    item_reward += alpha * gth_rerank[tid]["cids"][cid] / gth_cnt if gth_cnt else 0
            else:
                if relevance < 0:
                    item_reward += beta / gth_unhit_num["col"] if gth_unhit_num["col"] else 0

    if gth_cnt == 0:
        item_reward += alpha * 2
    if gth_unhit_num["row"] == 0:
        item_reward += beta
    if gth_unhit_num["col"] == 0:
        item_reward += beta
    final_reward = item_reward

    return final_reward

def reward_func_overall_tabrerank(completions, **kwargs):
    questions = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(question) for question in questions]
    questions = [
        question.split("<query>")[-1].split("</query>")[0].strip('\n') for question in questions
    ]

    tables = kwargs["tables"]
    descriptions = kwargs["table_description"]
    rerank_scores = [
        rerank_table_evidence(table, description, question)
        for table, description, question in zip(tables, descriptions, questions)
    ]
    table_evids = [kwargs["qa"][id]["table_evidence"] for id in range(len(kwargs["qa"]))]

    rerank_rewards = [
        get_rerank_overall_reward(rerank_score, table_evid)
        for rerank_score, table_evid in zip(rerank_scores, table_evids)
    ]

    rewards = [fr + rr for fr, rr in zip(format_rewards, rerank_rewards)]
    return rewards

def get_rerank_reward(pred_rerank_score, gth_evidence):
    gth_reward = 0
    gth_cnt = 0

    for item in gth_evidence:
        tid, rid, cid = item.split('-')
        tid = int(tid); rid = int(rid); cid = int(cid)
        try:
            gth_reward += pred_rerank_score[tid]["rids"][rid] ** 0.5
        except Exception:
            gth_reward += 1
        try:
            gth_reward += pred_rerank_score[tid]["cids"][cid] ** 0.5
        except Exception:
            gth_reward += 1
        gth_reward /= 2
        gth_cnt += 1

    final_reward = gth_reward / gth_cnt if gth_cnt != 0 else 1.0
    final_reward = reward_value(final_reward)

    return final_reward

def reward_func_tabrerank(completions, **kwargs):
    questions = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(question) for question in questions]
    questions = [
        question.split("<query>")[-1].split("</query>")[0].strip('\n') for question in questions
    ]

    tables = kwargs["tables"]
    descriptions = kwargs["table_description"]
    rerank_scores = [
        rerank_table_evidence(table, description, question)
        for table, description, question in zip(tables, descriptions, questions)
    ]
    table_evids = [kwargs["qa"][id]["table_evidence"] for id in range(len(kwargs["qa"]))]

    rerank_rewards = [
        get_rerank_reward(rerank_score, table_evid)
        for rerank_score, table_evid in zip(rerank_scores, table_evids)
    ]

    rewards = [fr + rr for fr, rr in zip(format_rewards, rerank_rewards)]
    return rewards

def reward_func_joint(completions, **kwargs):
    questions = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(question) for question in questions]
    questions = [
        question.split("<query>")[-1].split("</query>")[0].strip('\n') for question in questions
    ]
    text_docs = kwargs["paragraphs"]
    table_docs = kwargs["table_description"]
    text_evids = [kwargs["qa"][id]["text_evidence"] for id in range(len(kwargs["qa"]))]
    table_evids = [kwargs["qa"][id]["table_evidence"] for id in range(len(kwargs["qa"]))]
    text_scores = [
        retriever.eval(retriever.retrieve(question, text_doc), text_evid)[REWARD_TYPE] 
        for question, text_doc, text_evid in zip(questions, text_docs, text_evids)
    ]
    table_scores = [
        retriever.eval(retriever.retrieve(question, table_doc), table_evid)[REWARD_TYPE] 
        for question, table_doc, table_evid in zip(questions, table_docs, table_evids)
    ]
    retrieve_rewards = [
        reward_value(text_score) + reward_value(table_score)
        for text_score, table_score in zip(text_scores, table_scores)
    ]
    rewards = [fr + rr for fr, rr in zip(format_rewards, retrieve_rewards)]
    return rewards

def reward_func_text(completions, **kwargs):
    questions = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(question) for question in questions]
    questions = [
        question.split("<query>")[-1].split("</query>")[0].strip('\n') for question in questions
    ]
    text_docs = kwargs["paragraphs"]
    text_evids = [kwargs["qa"][id]["text_evidence"] for id in range(len(kwargs["qa"]))]
    text_scores = [
        retriever.eval(retriever.retrieve(question, text_doc), text_evid)[REWARD_TYPE] 
        for question, text_doc, text_evid in zip(questions, text_docs, text_evids)
    ]
    retrieve_rewards = [reward_value(text_score)*2 for text_score in text_scores]
    rewards = [fr + rr for fr, rr in zip(format_rewards, retrieve_rewards)]
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_type', type=str, default='joint', choices=['joint', 'text', 'tabrerank'])
    parser.add_argument('--recall', action='store_true')
    parser.add_argument('--overall', action='store_true')
    args = parser.parse_args()

    REWARD_TYPE = 1 if args.recall else 2
    train_data_path = 'datasets/multihiertt/train_new.json'
    retriever_model_path = 'models/Qwen3-Embedding-0.6B'
    augment_model_path = 'models/Qwen3-1.7B'
    reranker_model_path = 'models/Qwen3-Reranker-0.6B'
    save_model_path = f'models/HybTQA-{args.aug_type}-{REWARD_TYPE}'
    if args.overall:
        save_model_path += "-all"

    accelerator = Accelerator()

    dataset = load_dataset('json', data_files=train_data_path)

    dataset = dataset.map(make_conversation)
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=42)

    training_args = GRPOConfig(
        output_dir=save_model_path,
        learning_rate=5e-6,
        num_train_epochs=2,
        per_device_train_batch_size=32,
        max_completion_length=256,
        num_generations=8,
        max_prompt_length=128,
        logging_steps=5,
        save_steps=2000,
        report_to=None
    )

    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True
    )

    
    if args.aug_type == 'tabrerank':
        # reranker = Reranker(reranker_model_path)
        reranker_lambda = 0.05
        if args.overall:
            reward_func = reward_func_overall_tabrerank
        else:
            reward_func = reward_func_tabrerank
    else:
        retriever = Retriever(retriever_model_path)
        if args.aug_type == 'joint':
            reward_func = reward_func_joint
        elif args.aug_type == 'text':
            reward_func = reward_func_text

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)