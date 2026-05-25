import os
import json
import argparse
from dotenv import load_dotenv
from modules.retriever import *
from modules.agent import *


def ensure_dirs(*dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def init_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--batch", default=100, type=int)
    parser.add_argument("--max_concurrency", default=16, type=int)
    
    parser.add_argument("--data_root", default="./datasets/multihiertt")
    parser.add_argument("--save_root", default="./results/multihiertt")

    parser.add_argument("--retrieve_type", default="none", choices=["none", "dpr", "bm25", "hybrid", "gth"])

    parser.add_argument("--top_k", default=20, type=int)  # only for dpr
    parser.add_argument("--top_p", default=0.4, type=float)  # only for dpr, if top_p > 0, use top_p to filter tables
    parser.add_argument("--bm25_weight", default=0.5, type=float)  # only for hybrid
    parser.add_argument("--query_aug", default="none", choices=[
        "none", "raw_aug", "text-hard", "text-soft", "table-hard", "table-soft", 
        "text-table-hard", "text-table-soft", "text-hard-table-soft", "joint-hard", "joint-soft", 
        "hybrid", "joint-soft-0.1", "joint-soft-cont", "joint-soft-0.1-cont"
    ])  # only for dpr
    parser.add_argument("--stored_embs_path", default="./stored")  # only for dpr
    parser.add_argument("--device", default="cuda")  # only for dpr
    parser.add_argument("--tabform", action="store_true")  # only for dpr or gth

    parser.add_argument("--extend_header", action="store_true")  # only for raw_ext, grpo_ext

    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    return args


def init_dataset(args):
    data_root = args.data_root
    data_type = "dev" if args.dev else "test"
    data_file = f"{data_type}.json"
    data_path = os.path.join(data_root, data_file)

    with open(data_path, "r") as f:
        samples = json.loads(f.read())
        for i in range(len(samples)):
            samples[i]["id"] = i
    if args.retrieve_type in ["bm25", "hybrid"] and args.query_aug != "none":
        query_aug_path = os.path.join(data_root, f"{data_type}_{args.query_aug}.jsonl")
        with open(query_aug_path, "r") as f:
            augmented_queries = {
                item["uid"]: item["new_query"]
                for item in [json.loads(line) for line in f if line.strip()]
            }
        for sample in samples:
            sample["augmented_question"] = augmented_queries.get(sample["uid"], sample["qa"]["question"])
    if args.end == -1:
        args.end = len(samples)

    args.samples = samples


def init_model_config(args):
    load_dotenv()
    llm_model = os.getenv("LLM_MODEL")
    llm_config = {
        "llm_model": llm_model,
        "api_key": os.getenv('API_KEY'),
        "base_url": os.getenv('BASE_URL'),
    }

    args.llm_config = llm_config

    save_root = args.save_root
    save_root_model = os.path.join(save_root, llm_model.split('/')[-1])

    setting = "e2e"
    if args.retrieve_type != "none":
        setting = args.retrieve_type
        if args.retrieve_type in ["dpr", "bm25", "hybrid"]:
            setting += f"_top{args.top_k}"
            if args.top_p > 0:
                setting += f"_p{args.top_p}"
        if args.retrieve_type == "hybrid":
            setting += f"_bm25w{args.bm25_weight}"
        if args.retrieve_type in ["dpr", "bm25", "hybrid"]:
            if args.query_aug != "none":
                setting += f"_{args.query_aug}"
        if args.tabform:
            setting += "_tabform"
    if not args.dev:
        setting += "_test"

    save_root_setting = os.path.join(save_root_model, setting)
    print("Saving to: ", save_root_setting)

    ensure_dirs(save_root, save_root_model, save_root_setting)
    args.save_root_setting = save_root_setting


def init_models(args):
    retriever = None
    if args.retrieve_type == "dpr":
        retriever = DensePassageRetriever(args)
    elif args.retrieve_type == "bm25":
        retriever = BM25Retriever(args)
    elif args.retrieve_type == "hybrid":
        retriever = HybridRetriever(args)
    elif args.retrieve_type == "gth":
        retriever = GroundTruthRetriever(args)
    args.retriever = retriever

    agent = ThinkAgent(args)
    args.agent = agent


def init():
    args = init_args()
    init_dataset(args)
    init_model_config(args)
    init_models(args)

    return args
