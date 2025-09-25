import os
import json
import argparse
from dotenv import load_dotenv
from modules.retriever import *
from modules.agent import *


def ensure_dirs(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


def init_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--batch", default=100, type=int)
    
    parser.add_argument("--data_root", default="./datasets/multihiertt")
    parser.add_argument("--save_root", default="./results/multihiertt")

    parser.add_argument("--retrieve_type", default="none", choices=["none", "dpr", "gth"])

    parser.add_argument("--top_k", default=10, type=int)  # only for dpr
    parser.add_argument("--query_aug", default="none", choices=["none", "raw_aug", "text_aug", "grpo_aug"])  # only for dpr
    parser.add_argument("--stored_embs_path", default="./stored")  # only for dpr
    parser.add_argument("--tabform", action="store_true")  # only for dpr or gth
    parser.add_argument("--tabextract", action="store_true")  # only for dpr and tabform
    parser.add_argument("--tabextract_type", default="none", choices=["none", "raw_ext", "grpo_ext"])
    parser.add_argument("--tabfilter", action="store_true")  # only for dpr and tabform
    parser.add_argument("--tabfilter_type", default="none", choices=["none", "raw_fil", "grpo_fil"])
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
        if args.retrieve_type == "dpr":
            setting += f"_top{args.top_k}"
            if args.query_aug != "none":
                setting += f"_{args.query_aug}"
        if args.tabform:
            setting += "_tabform"
            if args.tabextract:
                setting += "_tabextract"
                if args.retrieve_type != "gth" and args.tabextract_type != "none":
                    setting += f"_{args.tabextract_type}"
                    if args.extend_header:
                        setting += "_extend"
            elif args.tabfilter:
                setting += "_tabfilter"
                if args.retrieve_type != "gth" and args.tabfilter_type != "none":
                    setting += f"_{args.tabfilter_type}"
                    if args.extend_header:
                        setting += "_extend"
    save_root_setting = os.path.join(save_root_model, setting)

    ensure_dirs(save_root, save_root_model, save_root_setting)
    args.save_root_setting = save_root_setting


def init_models(args):
    retriever = None
    if args.retrieve_type == "dpr":
        retriever = DensePassageRetriever(args)
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