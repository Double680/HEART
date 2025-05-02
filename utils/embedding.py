import os
import json
import asyncio
import torch
from tqdm import tqdm
from utils.api_query import emb_query
from utils.util import *
from openai import AsyncOpenAI


async def get_emb(args, emb_client, emb_model, inputs):
    emb = []
    batch = args.batch
    for i in range(0, len(inputs), batch):
        left, right = i, min(i+batch, len(inputs))
        results = await emb_query(emb_client, emb_model, inputs[left:right])
        results = [results.data[i].embedding for i in range(len(results.data))]
        emb.extend(results)
    return emb


async def prepare_emb(args, samples, config, stored_emb_dir):
    if args.dev:
        data_type = "dev"
    else:
        data_type = "test"
    stored_emb_subdir = os.path.join(stored_emb_dir, data_type)
    ensure_dirs(stored_emb_dir, stored_emb_subdir)    
    
    emb_client = AsyncOpenAI(base_url=config["base_url"], api_key=config["api_key"])
    emb_model = config["emb_model"]

    # check whether sample embeddings exist
    print("Prepare embeddings")
    for i in tqdm(range(len(samples[args.start:args.end]))):
        sample = samples[i]
        uid = sample["uid"]
        text_st, table_st, question = sample['paragraphs'], sample['table_description'], sample['qa']['question']
        emb_path = os.path.join(stored_emb_subdir, f'{uid}.json')
        try:
            with open(emb_path, 'r') as file:
                sample_emb_dict = json.loads(file.read())
            assert sample_emb_dict["text_st_embs"].size(0) == len(text_st)
            assert sample_emb_dict["table_st_embs"].size(0) == len(table_st)
            assert sample_emb_dict["question_emb"].size(0) == 1
        except Exception:
            text_st_embs = await get_emb(args, emb_client, emb_model, text_st)
            table_stv = [table_st[key] for key in table_st]
            table_st_embs = await get_emb(args, emb_client, emb_model, table_stv)
            question_emb = await get_emb(args, emb_client, emb_model, [question])
            sample_emb_dict = {
                "text_st_embs": text_st_embs,
                "table_st_embs": table_st_embs,
                "question_emb": question_emb
            }
            with open(emb_path, 'w') as file:
                file.write(json.dumps(sample_emb_dict))
