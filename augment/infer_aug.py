import argparse
import os
import sys

sys.path.append("./")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from augment.utils import (
    apply_chat_template,
    extract_query,
    get_rank,
    get_torch_device,
    get_world_size,
    init_distributed,
    is_main_process,
    iter_batches,
    load_json,
    load_jsonl,
    load_text,
    make_aug_messages,
    parse_torch_dtype,
    shard_dataset,
    wait_for_all_processes,
    write_jsonl,
)


def generate_batch(model, tokenizer, device, template, samples, args):
    texts = [
        apply_chat_template(tokenizer, make_aug_messages(sample, template))
        for _, sample in samples
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs.input_ids.shape[1]
    outputs = tokenizer.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
    )

    rows = []
    for (index, sample), output in zip(samples, outputs):
        rows.append({
            "index": index,
            "uid": sample["uid"],
            "new_query": extract_query(output.strip("\n")),
        })
    return rows


def load_model_and_tokenizer(args, device):
    dtype = parse_torch_dtype(args.torch_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def merge_part_files(output_path, part_paths):
    rows = []
    for path in part_paths:
        if not os.path.exists(path):
            continue
        rows.extend(load_jsonl(path))

    rows.sort(key=lambda row: row["index"])
    write_jsonl(output_path, [
        {"uid": row["uid"], "new_query": row["new_query"]}
        for row in rows
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate augmented queries.")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--path", type=str, default="models/Qwen3-1.7B")
    parser.add_argument("--data_root", type=str, default="./datasets/multihiertt")
    parser.add_argument("--prompt_template_path", type=str, default="augment/aug_template.txt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Debug on the first N samples.")
    parser.add_argument("--keep_part_files", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    init_distributed()

    rank = get_rank()
    world_size = get_world_size()
    device = get_torch_device()

    dataset_type = "dev" if args.dev else "test"
    source_path = os.path.join(args.data_root, f"{dataset_type}.json")
    output_path = os.path.join(args.data_root, f"{dataset_type}_{args.name}.jsonl")
    part_path = f"{output_path}.rank{rank}.tmp"

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to regenerate it.")

    dataset = load_json(source_path)
    if args.limit is not None:
        dataset = dataset[:args.limit]

    template = load_text(args.prompt_template_path)
    model, tokenizer = load_model_and_tokenizer(args, device)

    shard = shard_dataset(dataset, rank, world_size)
    if is_main_process():
        print(
            "[Aug infer] "
            f"dataset={source_path}, "
            f"output={output_path}, "
            f"world_size={world_size}, "
            f"batch_size={args.batch_size}",
            flush=True,
        )

    rows = []
    progress = tqdm(
        iter_batches(shard, args.batch_size),
        total=(len(shard) + args.batch_size - 1) // args.batch_size,
        disable=not is_main_process(),
        desc=f"{dataset_type} rank{rank}",
    )
    for batch in progress:
        rows.extend(generate_batch(model, tokenizer, device, template, batch, args))

    write_jsonl(part_path, rows)
    wait_for_all_processes()

    if is_main_process():
        part_paths = [f"{output_path}.rank{idx}.tmp" for idx in range(world_size)]
        merge_part_files(output_path, part_paths)
        if not args.keep_part_files:
            for path in part_paths:
                if os.path.exists(path):
                    os.remove(path)
        print(f"[Aug infer] wrote {output_path}", flush=True)

    wait_for_all_processes()


if __name__ == "__main__":
    main()
