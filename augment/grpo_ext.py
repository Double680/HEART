import sys
sys.path.append('./')

from accelerate import Accelerator
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
import json
import torch

from modules.process_tables import TableStructure

def make_conversation(sample):
    with open("augment/ext_template.txt", "r") as file:
        template = file.read()

    content = template.replace("<QUESTION>", sample["qa"]["question"])

    tid = int(sample["paragraphs"][1].split("## Table ")[-1].split(" ##")[0])
    table_tree = TableStructure(sample["table"], tid, sample["table_description"])
    table_instruction = sample["paragraphs"][0]
    content = content.replace("<INSTRUCTION>", table_instruction)

    table_content = sample["table"]
    content = content.replace("<TABLE>", table_content)

    table_row_headers = "Row Headers (ID: Name): \n" + table_tree.list_row_headers() + "\n"
    table_col_headers = "Column Headers (ID: Name): \n" + table_tree.list_col_headers() + "\n"
    table_headers = table_row_headers + table_col_headers
    content = content.replace("<HEADERS>", table_headers)

    prompt = {
        "prompt": [{
            "role": "user", 
            "content": content + '/no_think'
        }]
    }
    return prompt

def reward_value(value):
    return 2 * value
    
def format_reward(answer):
    reward = -2
    if "<answer>" in answer and "</answer>" in answer:
        reward += 1
    if len(answer.split("<answer>")[-1].split("</answer>")[0]) > 0:
        reward += 1

    return reward

def get_evid_pred(answer):
    subtables = answer.split("<answer>")[-1].split("</answer>")[0].strip('\n')
    rows = subtables.split("<rows>")[-1].split("</rows>")[0].strip().split(' | ')
    rids = []
    for row in rows:
        try:
            rids.append(int(row))
        except Exception:
            continue

    columns = subtables.split("<columns>")[-1].split("</columns>")[0].strip().split(' | ')
    cids = []
    for column in columns:
        try:
            cids.append(int(column))
        except Exception:
            continue

    pred = { "rids": rids, "cids": cids }

    return pred

def recall_eval(pred, gth):
    recall_cnt = 0
    total_cnt = 0
    for item in gth:
        _, rid, cid = item.split('-')
        rid = int(rid)
        cid = int(cid)

        if rid in pred["rids"] and cid in pred["cids"]:
            recall_cnt += 1

        total_cnt += 1

    recall_score = (recall_cnt - total_cnt) * 0.5

    return recall_score

def IoU_eval(pred, gth):
    gth_dic = { "rids": [], "cids": [] }
    for item in gth: #"0-2-8"
        _, rid, cid = item.split('-')
        rid = int(rid)
        cid = int(cid)

        if rid not in gth_dic["rids"]:
            gth_dic["rids"].append(rid)
        if cid not in gth_dic["cids"]:
            gth_dic["cids"].append(cid)

    pred_row_set = set(pred["rids"])
    gth_row_set = set(gth_dic["rids"])
    row_joint = len(pred_row_set.intersection(gth_row_set))
    row_union = len(pred_row_set.union(gth_row_set))

    pred_col_set = set(pred["cids"])
    gth_col_set = set(gth_dic["cids"])
    col_joint = len(pred_col_set.intersection(gth_col_set))
    col_union = len(pred_col_set.union(gth_col_set))
        
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
        IoU_eval(pred, gth) + recall_eval(pred, gth) for pred, gth in zip(table_evid_pred, table_evid_gth)
    ]

    extract_rewards = [reward_value(table_score) for table_score in table_scores]
    rewards = [fr + rr for fr, rr in zip(format_rewards, extract_rewards)]
    return rewards

if __name__ == "__main__":
    train_data_path = 'datasets/multihiertt/train_ext_hit.json'
    extract_model_path = 'models/Qwen3-1.7B'
    save_model_path = f'models/HybTQA-grpo-ext'

    accelerator = Accelerator()

    dataset = load_dataset('json', data_files=train_data_path)

    dataset = dataset.map(make_conversation)
    # dataset = dataset.remove_columns(["uid", "tables"])
    dataset = dataset["train"]

    training_args = GRPOConfig(
        output_dir=save_model_path,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        max_completion_length=256,
        num_generations=8,
        max_prompt_length=768,
        logging_steps=5,
        save_steps=2000,
        report_to=None
    )

    model = AutoModelForCausalLM.from_pretrained(
        extract_model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
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
