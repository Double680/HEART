import sys
sys.path.append('./')

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
import json
import torch

def make_conversation(example):
    with open("augment/ext_template.txt", "r") as file:
        template = file.read()

    content = template.replace("<QUESTION>", example["qa"]["question"])
    tabular_content = ""
    tid = 0
    for i in range(len(example["paragraphs"])):
        if example["paragraphs"][i] == f"## Table {tid} ##":
            tabular_content += f"Table {tid} - "
            tabular_content += example["paragraphs"][i-1]
            tabular_content += example["tables"][tid]
            tid += 1
    content = content.replace("<TABLES>", tabular_content)

    prompt = {
        "prompt": [{
            "role": "user", 
            "content": content + '/no_think'
        }]
    }
    return prompt

def reward_value(value):
    return 4 * value
    
def format_reward(answer):
    reward = -6
    if "<answer>" in answer and "</answer>" in answer:
        reward += 1
    if len(answer.split("<answer>")[-1].split("</answer>")[0]) > 0:
        reward += 1
    try:
        answer = json.loads(answer.split("<answer>")[-1].split("</answer>")[0].strip('\n'))
        reward += 4
    except Exception:
        pass

    return reward

def get_evid_pred(answer):
    try:
        output = json.loads(answer)
    except Exception:
        output = {}
    return output

def recall_eval(pred, gth):
    recall_cnt = 0
    total_cnt = 0
    for item in gth:
        tid, rid, cid = item.split('-')
        rid = int(rid)
        cid = int(cid)
        if tid in pred:
            tid_pred = pred[tid]
            if rid in tid_pred["rid"] and cid in tid_pred["cid"]:
                recall_cnt += 1
        total_cnt += 1

    recall_score = recall_cnt / total_cnt if total_cnt else 1.0

    return recall_score

def IoU_eval(pred, gth):
    gth_dic = {}
    for item in gth: #"0-2-8"
        tid, rid, cid = item.split('-')
        rid = int(rid)
        cid = int(cid)
        if tid not in gth_dic:
            gth_dic[tid] = {"rid": [], "cid": []}
        if rid not in gth_dic[tid]["rid"]:
            gth_dic[tid]["rid"].append(rid)
        if cid not in gth_dic[tid]["cid"]:
            gth_dic[tid]["cid"].append(cid)
    
    row_joint, row_union = 0, 0
    col_joint, col_union = 0, 0

    for key in pred.keys():
        if key not in gth_dic:
            gth_dic[key] = {"rid": [], "cid": []}
        
        pred_row_set = set(pred[key]["rid"])
        gth_row_set = set(gth_dic[key]["rid"])
        row_joint += len(pred_row_set.intersection(gth_row_set))
        row_union += len(pred_row_set.union(gth_row_set))

        pred_col_set = set(pred[key]["cid"])
        gth_col_set = set(gth_dic[key]["cid"])
        col_joint += len(pred_col_set.intersection(gth_col_set))
        col_union += len(pred_col_set.union(gth_col_set))
        
    row_score = row_joint / row_union if row_union else 1.0
    col_score = col_joint / col_union if col_union else 1.0
    
    IoU_score = (row_score + col_score) / 2
    return IoU_score


def reward_func(completions, **kwargs):
    answers = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(answer) for answer in answers]
    table_evid_pred = [
        get_evid_pred(answer.split("<answer>")[-1].split("</answer>")[0].strip('\n')) for answer in answers
    ]
    table_evid_gth = [list(set(kwargs["qa"][id]["table_evidence"])) for id in range(len(kwargs["qa"]))]

    table_scores = [
        IoU_eval(pred, gth) for pred, gth in zip(table_evid_pred, table_evid_gth)
    ]

    extract_rewards = [reward_value(table_score) for table_score in table_scores]
    rewards = [fr + rr for fr, rr in zip(format_rewards, extract_rewards)]
    return rewards

if __name__ == "__main__":
    train_data_path = 'datasets/multihiertt/train_new.json'
    extract_model_path = 'models/Qwen3-4B'
    save_model_path = f'models/HybTQA-tabextract'

    dataset = load_dataset('json', data_files=train_data_path)

    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(["uid", "tables"])
    dataset = dataset["train"]

    training_args = GRPOConfig(
        output_dir=save_model_path,
        learning_rate=5e-6,
        num_train_epochs=2,
        per_device_train_batch_size=32,
        max_completion_length=384,
        num_generations=8,
        max_prompt_length=1536,
        logging_steps=5,
        save_steps=2000,
        report_to=None
    )

    model = AutoModelForCausalLM.from_pretrained(
        extract_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    reward_func = reward_func

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)