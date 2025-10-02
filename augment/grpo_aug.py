import sys
sys.path.append('./')

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from augment.retriever import Retriever
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


def process_text_scores(questions, **kwargs):
    text_docs = kwargs["paragraphs"]
    text_evids = [kwargs["qa"][id]["text_evidence"] for id in range(len(kwargs["qa"]))]
    if AUG_SOFT == 'soft' and AUG_TYPE != 'hybrid':
        text_scores = [
            retriever.soft_retrieve_eval(question, text_doc, text_evid, contrastive=CONTRASTIVE)
            for question, text_doc, text_evid in zip(questions, text_docs, text_evids)
        ]
    else:
        text_scores = [
            retriever.eval(retriever.retrieve(question, text_doc, top_k=20), text_evid)[1] 
            for question, text_doc, text_evid in zip(questions, text_docs, text_evids)
        ]
    return text_scores


def process_table_scores(questions, **kwargs):
    table_docs = kwargs["table_description"]
    table_evids = [kwargs["qa"][id]["table_evidence_id"] for id in range(len(kwargs["qa"]))]
    if AUG_SOFT == 'soft' or AUG_TYPE == 'hybrid':
        table_scores = [
            retriever.soft_retrieve_eval(question, table_doc, table_evid, contrastive=CONTRASTIVE)
            for question, table_doc, table_evid in zip(questions, table_docs, table_evids)
        ]
    else:
        table_scores = [
            retriever.eval(retriever.retrieve(question, table_doc, top_k=20), table_evid)[1] 
            for question, table_doc, table_evid in zip(questions, table_docs, table_evids)
        ]
    return table_scores


def reward_func(completions, **kwargs):
    questions = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(question) for question in questions]
    questions = [
        question.split("<query>")[-1].split("</query>")[0].strip('\n') for question in questions
    ]

    if AUG_TYPE == "text":
        text_scores = process_text_scores(questions, **kwargs)
        table_scores = text_scores
    elif AUG_TYPE == "table":
        table_scores = process_table_scores(questions, **kwargs)
        text_scores = table_scores
    else:
        text_scores = process_text_scores(questions, **kwargs)
        table_scores = process_table_scores(questions, **kwargs)

    retrieve_rewards = [
        reward_value(text_score) + reward_value(table_score)
        for text_score, table_score in zip(text_scores, table_scores)
    ]

    rewards = [fr + rr for fr, rr in zip(format_rewards, retrieve_rewards)]
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_type', type=str, default='joint', choices=['joint', 'text', 'table', 'hybrid'])  # hybrid: text-hard, table-soft
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--contrastive', action='store_true')
    args = parser.parse_args()

    AUG_SOFT = 'soft' if args.soft else 'hard'
    CONTRASTIVE = args.contrastive
    AUG_TYPE = args.aug_type

    train_data_path = 'datasets/multihiertt/train_new.json'
    retriever_model_path = 'models/Qwen3-Embedding-0.6B'
    augment_model_path = 'models/Qwen3-1.7B'
    save_model_path = f'models/{args.aug_type}-{AUG_SOFT}'
    if args.beta > 0:
        save_model_path += f'-beta{args.beta}'
    if CONTRASTIVE:
        save_model_path += '-cont'

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
        beta=args.beta,
        report_to=None
    )

    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True
    )

    retriever = Retriever(retriever_model_path)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)