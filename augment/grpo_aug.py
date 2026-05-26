import argparse
import os
import re
import sys
from dataclasses import dataclass

sys.path.append("./")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from modules.retriever import build_evidence_ranker


NO_THINK_SUFFIX = " /no_think"
QUERY_PATTERN = re.compile(r"^\s*<query>(.*?)</query>\s*$", re.DOTALL)
RETRIEVAL_REWARD_SCALE = 4
DEFAULT_RETRIEVAL_TOP_K = 20
EVIDENCE_FIELDS = {
    "text": ("paragraphs", "text_evidence"),
    "table": ("table_description", "table_evidence_id"),
}


@dataclass
class RewardConfig:
    aug_type: str
    use_soft_reward: bool
    metric_log_steps: int
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K


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


def mean(values):
    return sum(values) / len(values) if values else 0.0


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


def reward_value(value):
    return RETRIEVAL_REWARD_SCALE * value


def get_save_model_path(args):
    if args.aug_type == "hybrid":
        save_model_path = "models/hybrid"
    else:
        aug_soft = "soft" if args.soft else "hard"
        save_model_path = f"models/{args.aug_type}-{aug_soft}"

    if args.beta > 0:
        save_model_path += f"-beta{args.beta}"
    if args.reward_retrieve_type not in ["dense", "dpr"]:
        save_model_path += f"-reward-{args.reward_retrieve_type}"
        if args.reward_retrieve_type == "hybrid":
            save_model_path += f"-bm25w{args.reward_bm25_weight}"
    return save_model_path


def extract_query(output):
    match = QUERY_PATTERN.match(output)
    if match:
        return match.group(1).strip()
    return output.split("<query>")[-1].split("</query>")[0].strip("\n").strip()


def format_reward(output):
    match = QUERY_PATTERN.match(output)
    if match and match.group(1).strip():
        return 0
    if "<query>" in output and "</query>" in output:
        return -1
    return -2


class RetrievalReward:
    def __init__(self, retriever, config):
        self.retriever = retriever
        self.config = config
        self.calls = 0

    def compute(self, completions, **kwargs):
        raw_outputs = [
            completion[0]["content"].split("</think>")[-1].strip("\n")
            for completion in completions
        ]
        format_rewards = [format_reward(output) for output in raw_outputs]
        queries = [extract_query(output) for output in raw_outputs]

        text_scores, table_scores = self._score_by_aug_type(queries, kwargs)

        self.calls += 1
        if self.config.metric_log_steps > 0 and self.calls % self.config.metric_log_steps == 0:
            self._log_metrics(queries, text_scores, table_scores, format_rewards, kwargs)

        retrieve_rewards = [
            reward_value(text_score) + reward_value(table_score)
            for text_score, table_score in zip(text_scores, table_scores)
        ]
        return [fr + rr for fr, rr in zip(format_rewards, retrieve_rewards)]

    def _score_by_aug_type(self, queries, batch):
        if self.config.aug_type == "text":
            scores = self._score_evidence(queries, batch, "text")
            return scores, scores
        if self.config.aug_type == "table":
            scores = self._score_evidence(queries, batch, "table")
            return scores, scores
        return (
            self._score_evidence(queries, batch, "text"),
            self._score_evidence(queries, batch, "table"),
        )

    def _use_soft_reward(self, evidence_type):
        if self.config.aug_type == "hybrid":
            return evidence_type == "table"
        return self.config.use_soft_reward

    def _batch_evidence(self, batch, evidence_type):
        docs_key, evid_key = EVIDENCE_FIELDS[evidence_type]
        return batch[docs_key], [qa[evid_key] for qa in batch["qa"]]

    def _score_evidence(self, queries, batch, evidence_type):
        docs_list, evids_list = self._batch_evidence(batch, evidence_type)
        if self._use_soft_reward(evidence_type):
            return [
                self.retriever.soft_retrieve_eval(query, docs, evids)
                for query, docs, evids in zip(queries, docs_list, evids_list)
            ]

        return [
            self.retriever.eval(self.retriever.retrieve(query, docs, top_k=self.config.retrieval_top_k), evids)[1]
            for query, docs, evids in zip(queries, docs_list, evids_list)
        ]

    def _retrieval_metrics(self, queries, docs_list, evids_list):
        precisions, recalls, ndcgs = [], [], []

        for query, docs, evids in zip(queries, docs_list, evids_list):
            if not docs:
                empty_score = 1.0 if not evids else 0.0
                precisions.append(empty_score)
                recalls.append(empty_score)
                ndcgs.append(empty_score)
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

    def _log_metrics(self, queries, text_scores, table_scores, format_rewards, batch):
        if not is_main_process():
            return

        text_docs, text_evids = self._batch_evidence(batch, "text")
        table_docs, table_evids = self._batch_evidence(batch, "table")
        text_metrics = self._retrieval_metrics(
            queries,
            text_docs,
            text_evids,
        )
        table_metrics = self._retrieval_metrics(
            queries,
            table_docs,
            table_evids,
        )
        valid_format = sum(1 for reward in format_rewards if reward == 0) / len(format_rewards)

        print(
            "[GRPO retrieval] "
            f"calls={self.calls} "
            f"text_p={text_metrics['precision']:.4f} "
            f"text_r={text_metrics['recall']:.4f} "
            f"text_ndcg={text_metrics['ndcg']:.4f} "
            f"table_p={table_metrics['precision']:.4f} "
            f"table_r={table_metrics['recall']:.4f} "
            f"table_ndcg={table_metrics['ndcg']:.4f} "
            f"text_reward={mean(text_scores):.4f} "
            f"table_reward={mean(table_scores):.4f} "
            f"format_valid={valid_format:.4f}",
            flush=True,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Train the query augmentor with GRPO.")

    data_group = parser.add_argument_group("Data and model paths")
    data_group.add_argument("--train_data_path", type=str, default="datasets/multihiertt/train_new.json")
    data_group.add_argument("--prompt_template_path", type=str, default="augment/aug_template.txt")
    data_group.add_argument("--augment_model_path", type=str, default="models/Qwen3-1.7B")
    data_group.add_argument("--retriever_model_path", type=str, default="models/Qwen3-Embedding-0.6B")

    reward_group = parser.add_argument_group("Augmentation and reward")
    reward_group.add_argument(
        "--aug_type",
        type=str,
        default="hybrid",
        choices=["joint", "text", "table", "hybrid"],
        help="joint: text+table; hybrid: text hard reward + table soft reward.",
    )
    reward_group.add_argument("--soft", action="store_true", help="Use soft retrieval reward for non-hybrid modes.")
    reward_group.add_argument("--retrieval_top_k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
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

    return parser


def build_grpo_config(args):
    return GRPOConfig(
        output_dir=get_save_model_path(args),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        beta=args.beta,
        report_to=None,
    )


def build_augment_model(model_path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )


def build_reward_evaluator(args, local_device):
    retriever_device = resolve_retriever_device(args.retriever_device, local_device)
    if is_main_process():
        print(
            "[GRPO train] "
            f"local_device={local_device}, "
            f"reward_retrieve_type={args.reward_retrieve_type}, "
            f"retriever_device={retriever_device}",
            flush=True,
        )

    retriever = build_evidence_ranker(
        args.reward_retrieve_type,
        model_name=args.retriever_model_path,
        device=retriever_device,
        bm25_weight=args.reward_bm25_weight,
    )
    reward_config = RewardConfig(
        aug_type=args.aug_type,
        use_soft_reward=args.soft,
        metric_log_steps=args.metric_log_steps,
        retrieval_top_k=args.retrieval_top_k,
    )
    return RetrievalReward(retriever, reward_config)


def main():
    args = build_parser().parse_args()
    local_device = get_local_device()

    dataset = load_train_dataset(args.train_data_path, args.prompt_template_path, args.seed)
    training_args = build_grpo_config(args)
    model = build_augment_model(args.augment_model_path)
    reward_evaluator = build_reward_evaluator(args, local_device)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_evaluator.compute,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
