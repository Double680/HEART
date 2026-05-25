import sys
sys.path.append('./')

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from augment.retriever import Retriever
from modules.process_tables import *
from transformers import AutoModelForCausalLM
import torch
import argparse
import re
import os

NO_THINK_SUFFIX = " /no_think"
QUERY_PATTERN = re.compile(r"^\s*<query>(.*?)</query>\s*$", re.DOTALL)
REWARD_CALLS = 0
METRIC_LOG_STEPS = 10


def get_rank():
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_rank() == 0


def get_local_device():
    local_rank = os.environ.get("LOCAL_RANK")
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(int(local_rank))
            return f"cuda:{local_rank}"
        return "cuda"
    return "cpu"


def resolve_retriever_device(device, local_device):
    if device == "auto":
        return local_device
    return device


def make_conversation(example):
    with open("augment/aug_template.txt", "r") as file:
        template = file.read()

    prompt = {
        "prompt": [{
            "role": "user", 
            "content": template.replace("<QUESTION>", example["qa"]["question"]) + NO_THINK_SUFFIX
        }]
    }
    return prompt


def reward_value(value):
    return 4 * value


def get_save_model_path(args):
    if args.aug_type == 'hybrid':
        save_model_path = 'models/hybrid'
    else:
        aug_soft = 'soft' if args.soft else 'hard'
        save_model_path = f'models/{args.aug_type}-{aug_soft}'
    if args.beta > 0:
        save_model_path += f'-beta{args.beta}'
    if args.contrastive:
        save_model_path += '-cont'
    return save_model_path
    

def extract_query(output):
    match = QUERY_PATTERN.match(output)
    if match:
        return match.group(1).strip()
    return output.split("<query>")[-1].split("</query>")[0].strip('\n').strip()


def format_reward(output):
    match = QUERY_PATTERN.match(output)
    if match and match.group(1).strip():
        return 0
    if "<query>" in output and "</query>" in output:
        return -1
    return -2


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


def mean(values):
    return sum(values) / len(values) if values else 0.0


def process_retrieval_metrics(questions, docs_list, evids_list, top_k=20):
    precisions, recalls, ndcgs = [], [], []
    for question, docs, evids in zip(questions, docs_list, evids_list):
        if not docs:
            precisions.append(1.0 if not evids else 0.0)
            recalls.append(1.0 if not evids else 0.0)
            ndcgs.append(1.0 if not evids else 0.0)
            continue
        precision, recall, ndcg = retriever.eval(
            retriever.retrieve(question, docs, top_k=top_k),
            evids
        )
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    return {
        "precision": mean(precisions),
        "recall": mean(recalls),
        "ndcg": mean(ndcgs),
    }


def log_retrieval_metrics(raw_outputs, queries, text_scores, table_scores, format_rewards, **kwargs):
    if not is_main_process():
        return

    text_docs = kwargs["paragraphs"]
    text_evids = [kwargs["qa"][id]["text_evidence"] for id in range(len(kwargs["qa"]))]
    table_docs = kwargs["table_description"]
    table_evids = [kwargs["qa"][id]["table_evidence_id"] for id in range(len(kwargs["qa"]))]

    text_metrics = process_retrieval_metrics(queries, text_docs, text_evids, top_k=20)
    table_metrics = process_retrieval_metrics(queries, table_docs, table_evids, top_k=20)
    valid_format = sum(1 for reward in format_rewards if reward == 0) / len(format_rewards) if format_rewards else 0.0

    print(
        "[GRPO retrieval] "
        f"calls={REWARD_CALLS} "
        f"text_p={text_metrics['precision']:.4f} "
        f"text_r={text_metrics['recall']:.4f} "
        f"text_ndcg={text_metrics['ndcg']:.4f} "
        f"table_p={table_metrics['precision']:.4f} "
        f"table_r={table_metrics['recall']:.4f} "
        f"table_ndcg={table_metrics['ndcg']:.4f} "
        f"text_reward={mean(text_scores):.4f} "
        f"table_reward={mean(table_scores):.4f} "
        f"format_valid={valid_format:.4f}",
        flush=True
    )


def reward_func(completions, **kwargs):
    global REWARD_CALLS

    raw_outputs = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(output) for output in raw_outputs]
    queries = [extract_query(output) for output in raw_outputs]

    if AUG_TYPE == "text":
        text_scores = process_text_scores(queries, **kwargs)
        table_scores = text_scores
    elif AUG_TYPE == "table":
        table_scores = process_table_scores(queries, **kwargs)
        text_scores = table_scores
    else:
        text_scores = process_text_scores(queries, **kwargs)
        table_scores = process_table_scores(queries, **kwargs)

    REWARD_CALLS += 1
    if METRIC_LOG_STEPS > 0 and REWARD_CALLS % METRIC_LOG_STEPS == 0:
        log_retrieval_metrics(raw_outputs, queries, text_scores, table_scores, format_rewards, **kwargs)

    retrieve_rewards = [
        reward_value(text_score) + reward_value(table_score)
        for text_score, table_score in zip(text_scores, table_scores)
    ]

    rewards = [fr + rr for fr, rr in zip(format_rewards, retrieve_rewards)]
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_type', type=str, default='hybrid', choices=['joint', 'text', 'table', 'hybrid'])  # hybrid: text-hard, table-soft
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--train_data_path', type=str, default='datasets/multihiertt/train_new.json')
    parser.add_argument('--retriever_model_path', type=str, default='models/Qwen3-Embedding-0.6B')
    parser.add_argument('--augment_model_path', type=str, default='models/Qwen3-1.7B')
    parser.add_argument('--retriever_device', type=str, default='auto')
    parser.add_argument('--metric_log_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_completion_length', type=int, default=256)
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=2000)
    args = parser.parse_args()

    local_device = get_local_device()
    AUG_SOFT = 'soft' if args.soft else 'hard'
    CONTRASTIVE = args.contrastive
    AUG_TYPE = args.aug_type
    METRIC_LOG_STEPS = args.metric_log_steps

    train_data_path = args.train_data_path
    retriever_model_path = args.retriever_model_path
    augment_model_path = args.augment_model_path
    save_model_path = get_save_model_path(args)

    dataset = load_dataset('json', data_files=train_data_path)

    dataset = dataset.map(make_conversation)
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=42)

    training_args = GRPOConfig(
        output_dir=save_model_path,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        beta=args.beta,
        report_to=None
    )

    model = AutoModelForCausalLM.from_pretrained(
        augment_model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True
    )

    retriever_device = resolve_retriever_device(args.retriever_device, local_device)
    if is_main_process():
        print(f"[GRPO train] local_device={local_device}, retriever_device={retriever_device}", flush=True)
    retriever = Retriever(retriever_model_path, device=retriever_device)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
