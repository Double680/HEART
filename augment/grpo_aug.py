import argparse
import sys
from dataclasses import dataclass

sys.path.append("./")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from augment.utils import (
    DEFAULT_RETRIEVAL_TOP_K,
    EVIDENCE_FIELDS,
    NO_THINK_SUFFIX,
    REWARD_AGGREGATIONS,
    REWARD_TYPES,
    enabled_evidence_types,
    extract_query,
    get_local_device,
    is_main_process,
    mean,
    resolve_retriever_device,
    std,
)
from modules.retriever import build_evidence_ranker


@dataclass
class RewardConfig:
    text_reward: str
    table_reward: str
    metric_log_steps: int
    reward_aggregation: str
    num_generations: int
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K


def load_prompt_template(template_path):
    with open(template_path, "r") as file:
        return file.read()


def make_conversation(example, template):
    return {
        "prompt": [{
            "role": "user",
            "content": template.replace("<QUESTION>", example["qa"]["question"]) + NO_THINK_SUFFIX,
        }]
    }


def load_train_dataset(train_data_path, prompt_template_path, seed):
    template = load_prompt_template(prompt_template_path)
    dataset = load_dataset("json", data_files=train_data_path)["train"]
    dataset = dataset.map(make_conversation, fn_kwargs={"template": template})
    return dataset.shuffle(seed=seed)


def get_save_model_path(args):
    if args.text_reward == "hard" and args.table_reward == "soft":
        save_model_path = "models/hybrid"
    else:
        reward_parts = [
            f"{evidence_type}-{reward_type}"
            for evidence_type, reward_type in [
                ("text", args.text_reward),
                ("table", args.table_reward),
            ]
            if reward_type != "none"
        ]
        save_model_path = f"models/{'-'.join(reward_parts)}"

    if args.beta > 0:
        save_model_path += f"-beta{args.beta}"
    if args.reward_aggregation == "advantage_then_sum":
        save_model_path += "-advsum"
    if args.reward_retrieve_type not in ["dense", "dpr"]:
        save_model_path += f"-reward-{args.reward_retrieve_type}"
        if args.reward_retrieve_type == "hybrid":
            save_model_path += f"-bm25w{args.reward_bm25_weight}"
    return save_model_path


class RetrievalReward:
    def __init__(self, retriever, config):
        self.retriever = retriever
        self.config = config
        self.calls = {evidence_type: 0 for evidence_type in EVIDENCE_FIELDS}

    def reward_funcs(self):
        reward_funcs = []
        if self.config.text_reward != "none":
            reward_funcs.append(self.text_retrieval_reward)
        if self.config.table_reward != "none":
            reward_funcs.append(self.table_retrieval_reward)
        return reward_funcs

    def text_retrieval_reward(self, completions, **batch):
        return self.retrieval_reward(completions, batch, "text")

    def table_retrieval_reward(self, completions, **batch):
        return self.retrieval_reward(completions, batch, "table")

    def retrieval_reward(self, completions, batch, evidence_type):
        queries = [extract_query(self._content(completion)) for completion in completions]
        scores = self._score_evidence(queries, batch, evidence_type)

        self.calls[evidence_type] += 1
        if (
            self.config.metric_log_steps > 0
            and self.calls[evidence_type] % self.config.metric_log_steps == 0
        ):
            self._log_metrics(queries, scores, batch, evidence_type)

        return self._aggregate_reward_values(scores)

    def _content(self, completion):
        return completion[0]["content"].split("</think>")[-1].strip("\n")

    def _aggregate_reward_values(self, rewards):
        if self.config.reward_aggregation == "sum_then_advantage":
            return rewards
        return self._group_advantages(rewards)

    def _group_advantages(self, rewards):
        advantages = [None] * len(rewards)
        group_size = self.config.num_generations

        for start in range(0, len(rewards), group_size):
            end = min(start + group_size, len(rewards))
            valid_indices = [idx for idx in range(start, end) if rewards[idx] is not None]
            if not valid_indices:
                continue

            valid_rewards = [rewards[idx] for idx in valid_indices]
            group_mean = mean(valid_rewards)
            group_std = std(valid_rewards)
            for idx in valid_indices:
                advantages[idx] = 0.0 if group_std == 0 else (rewards[idx] - group_mean) / group_std

        return advantages

    def _reward_type(self, evidence_type):
        return getattr(self.config, f"{evidence_type}_reward")

    def _batch_evidence(self, batch, evidence_type):
        docs_key, evid_key = EVIDENCE_FIELDS[evidence_type]
        return batch[docs_key], [qa[evid_key] for qa in batch["qa"]]

    def _score_evidence(self, queries, batch, evidence_type):
        docs_list, evids_list = self._batch_evidence(batch, evidence_type)

        scores = []
        for query, docs, evids in zip(queries, docs_list, evids_list):
            if not evids:
                scores.append(None)
                continue

            if self._reward_type(evidence_type) == "soft":
                scores.append(self.retriever.soft_retrieve_eval(query, docs, evids))
            else:
                retrieved = self.retriever.retrieve(
                    query,
                    docs,
                    top_k=self.config.retrieval_top_k,
                )
                scores.append(self.retriever.eval(retrieved, evids)[1])
        return scores

    def _retrieval_metrics(self, queries, docs_list, evids_list):
        precisions, recalls, ndcgs = [], [], []

        for query, docs, evids in zip(queries, docs_list, evids_list):
            if not evids:
                continue

            precision, recall, ndcg = self.retriever.eval(
                self.retriever.retrieve(query, docs, top_k=self.config.retrieval_top_k),
                evids,
            )
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        return {
            "precision": mean(precisions),
            "recall": mean(recalls),
            "ndcg": mean(ndcgs),
        }

    def _log_metrics(self, queries, evidence_scores, batch, evidence_type):
        if not is_main_process():
            return

        docs, evids = self._batch_evidence(batch, evidence_type)
        metrics = self._retrieval_metrics(queries, docs, evids)
        valid_scores = [score for score in evidence_scores if score is not None]
        labeled_ratio = len(valid_scores) / len(evidence_scores) if evidence_scores else 0.0

        print(
            "[GRPO retrieval] "
            f"calls={self.calls[evidence_type]} "
            f"type={evidence_type} "
            f"p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f} "
            f"ndcg={metrics['ndcg']:.4f} "
            f"reward={mean(valid_scores):.4f} "
            f"labeled_ratio={labeled_ratio:.4f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the query augmentor with GRPO.")

    data_group = parser.add_argument_group("Data and model paths")
    data_group.add_argument("--train_data_path", type=str, default="datasets/multihiertt/train_new.json")
    data_group.add_argument("--prompt_template_path", type=str, default="augment/aug_template.txt")
    data_group.add_argument("--augment_model_path", type=str, default="models/Qwen3-1.7B")
    data_group.add_argument("--retriever_model_path", type=str, default="models/Qwen3-Embedding-0.6B")

    reward_group = parser.add_argument_group("Augmentation and reward")
    reward_group.add_argument(
        "--text_reward",
        type=str,
        default="hard",
        choices=REWARD_TYPES,
        help="Reward type for text evidence.",
    )
    reward_group.add_argument(
        "--table_reward",
        type=str,
        default="soft",
        choices=REWARD_TYPES,
        help="Reward type for table evidence.",
    )
    reward_group.add_argument("--retrieval_top_k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    reward_group.add_argument(
        "--reward_aggregation",
        type=str,
        default="sum_then_advantage",
        choices=REWARD_AGGREGATIONS,
        help="sum_then_advantage keeps TRL's default behavior; advantage_then_sum normalizes each reward first.",
    )
    reward_group.add_argument("--text_weight", type=float, default=1.0)
    reward_group.add_argument("--table_weight", type=float, default=1.0)
    reward_group.add_argument(
        "--reward_retrieve_type",
        type=str,
        default="dense",
        choices=["dense", "dpr", "bm25", "hybrid"],
        help="Retriever used by the GRPO reward evaluator.",
    )
    reward_group.add_argument("--reward_bm25_weight", type=float, default=0.5, help="BM25 weight for hybrid reward.")

    grpo_group = parser.add_argument_group("GRPO training")
    grpo_group.add_argument("--learning_rate", type=float, default=5e-6)
    grpo_group.add_argument("--num_train_epochs", type=int, default=2)
    grpo_group.add_argument("--per_device_train_batch_size", type=int, default=32)
    grpo_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    grpo_group.add_argument("--max_completion_length", type=int, default=256)
    grpo_group.add_argument("--num_generations", type=int, default=8)
    grpo_group.add_argument("--beta", default=0, type=float)
    grpo_group.add_argument("--logging_steps", type=int, default=5)
    grpo_group.add_argument("--save_steps", type=int, default=2000)

    runtime_group = parser.add_argument_group("Runtime and logging")
    runtime_group.add_argument("--retriever_device", type=str, default="auto", help="auto, cpu, cuda, or cuda:N.")
    runtime_group.add_argument("--metric_log_steps", type=int, default=10)
    runtime_group.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.text_reward == "none" and args.table_reward == "none":
        raise ValueError("At least one of --text_reward or --table_reward must be hard or soft.")
    return args


def main():
    args = parse_args()
    local_device = get_local_device()
    retriever_device = resolve_retriever_device(args.retriever_device, local_device)

    if is_main_process():
        print(
            "[GRPO train] "
            f"local_device={local_device}, "
            f"reward_retrieve_type={args.reward_retrieve_type}, "
            f"retriever_device={retriever_device}",
            flush=True,
        )

    dataset = load_train_dataset(args.train_data_path, args.prompt_template_path, args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.augment_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    retriever = build_evidence_ranker(
        args.reward_retrieve_type,
        model_name=args.retriever_model_path,
        device=retriever_device,
        bm25_weight=args.reward_bm25_weight,
    )
    reward_evaluator = RetrievalReward(
        retriever,
        RewardConfig(
            text_reward=args.text_reward,
            table_reward=args.table_reward,
            metric_log_steps=args.metric_log_steps,
            reward_aggregation=args.reward_aggregation,
            num_generations=args.num_generations,
            retrieval_top_k=args.retrieval_top_k,
        ),
    )

    training_config = {
        "output_dir": get_save_model_path(args),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "beta": args.beta,
        "reward_weights": [
            getattr(args, f"{evidence_type}_weight")
            for evidence_type in enabled_evidence_types(args)
        ],
        "report_to": None,
    }
    if args.reward_aggregation == "advantage_then_sum":
        training_config["scale_rewards"] = False
    training_args = GRPOConfig(**training_config)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_evaluator.reward_funcs(),
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
