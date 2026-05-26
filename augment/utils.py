import os
import re

import torch


NO_THINK_SUFFIX = " /no_think"
QUERY_PATTERN = re.compile(r"^\s*<query>(.*?)</query>\s*$", re.DOTALL)
DEFAULT_RETRIEVAL_TOP_K = 10
REWARD_TYPES = ["none", "hard", "soft"]
REWARD_AGGREGATIONS = ["sum_then_advantage", "advantage_then_sum"]
EVIDENCE_FIELDS = {
    "text": ("paragraphs", "text_evidence"),
    "table": ("table_description", "table_evidence_id"),
}


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
    return local_device if device == "auto" else device


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def enabled_evidence_types(source):
    return [
        evidence_type
        for evidence_type in EVIDENCE_FIELDS
        if getattr(source, f"{evidence_type}_reward") != "none"
    ]


def extract_query(output):
    match = QUERY_PATTERN.match(output)
    if match:
        return match.group(1).strip()
    return output.split("<query>")[-1].split("</query>")[0].strip("\n").strip()
