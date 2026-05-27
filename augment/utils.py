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


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process():
    return get_rank() == 0


def get_local_device():
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
        return "cuda"
    return "cpu"


def get_torch_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def init_distributed():
    if get_world_size() == 1:
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is required for multi-process execution.")
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)


def wait_for_all_processes():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def resolve_retriever_device(device, local_device):
    return local_device if device == "auto" else device


def parse_torch_dtype(dtype_name, device):
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return getattr(torch, dtype_name)


def load_json(path):
    import json

    with open(path, "r") as file:
        return json.load(file)


def load_text(path):
    with open(path, "r") as file:
        return file.read()


def write_jsonl(path, rows):
    import json

    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def load_jsonl(path):
    import json

    with open(path, "r") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_aug_messages(example, template):
    return [{
        "role": "user",
        "content": template.replace("<QUESTION>", example["qa"]["question"]) + NO_THINK_SUFFIX,
    }]


def make_prompt_example(example, template):
    return {"prompt": make_aug_messages(example, template)}


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def shard_dataset(dataset, rank, world_size):
    return [
        (index, sample)
        for index, sample in enumerate(dataset)
        if index % world_size == rank
    ]


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
