import os
import json
import math
import argparse
from baselines.base_agent import *
from baselines.retriever import *
from utils.evaluate import *
# from utils.embedding import *
from utils.util import *
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import asyncio


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--agent', default='e2e')
    parser.add_argument('--aug', default='none', choices=['none', 'raw_aug', 'grpo_aug'], type=str)
    parser.add_argument('--retriever', default='none')
    parser.add_argument('--retrieve_k', default=10, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--batch', default=100, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--think', action='store_true')
    parser.add_argument('--tabheader', action='store_true')
    args = parser.parse_args()

    # load data
    dataset_type = "dev" if args.dev else "test"
    dataset_root = './datasets/multihiertt'
    input_path = f'{dataset_root}/{dataset_type}.json'
    with open(input_path, 'r') as file:
        samples = json.loads(file.read())
        for i in range(len(samples)):
            samples[i]['id'] = i
    if args.end == -1:
        args.end = len(samples)

    if args.tabheader:
        table_header_path = f'{dataset_root}/{dataset_type}_headers.json'
        with open(table_header_path, 'r') as file:
            table_headers = json.loads(file.read())
        for i in range(len(samples)):
            uid = samples[i]['uid']
            samples[i]['table_headers'] = table_headers[uid]
            table_header_ids = {}
            for key in samples[i]['table_description']:
                tid, row, col = key.split('-')
                tid, row, col = int(tid), int(row), int(col)
                if tid not in table_header_ids:
                    table_header_ids[tid] = {'row': 10000, 'col': 10000}
                table_header_ids[tid]['row'] = min(table_header_ids[tid]['row'], row-1)
                table_header_ids[tid]['col'] = min(table_header_ids[tid]['col'], col-1)
            samples[i]['table_headers_max_ids'] = table_header_ids

    # load models
    load_dotenv()
    llm_config = {
        "llm_model": os.getenv('LLM_MODEL'),
        "api_key": os.getenv('API_KEY'),
        "base_url": os.getenv('BASE_URL'),
        "think_mode": args.think
    }

    result_path_dir = f'./results'
    result_path_model = os.path.join(result_path_dir, llm_config["llm_model"].split('/')[-1])
    result_path_agent = os.path.join(result_path_model, f'{args.agent}-{args.retriever}-{args.aug}-{args.retrieve_k}')
    if args.tabheader:
        result_path_agent = f'{result_path_agent}-tabheader'
    if args.dev:
        result_path_root = os.path.join(result_path_agent, 'dev')
    else:
        result_path_root = os.path.join(result_path_agent, 'test')
    
    ensure_dirs(result_path_dir, result_path_model, result_path_agent, result_path_root)
    args.result_path_root = result_path_root

    retriever = None
    if args.retriever != "none":
        stored_emb_dir = './stored'
        args.stored_emb_dir = stored_emb_dir

        if args.retriever == "dpr":
            retriever = DensePassageRetriever(args.stored_emb_dir, args.aug, args.tabheader, top_k=args.retrieve_k, gpu=args.gpu)
        elif args.retriever == "gth":
            retriever = GroundTruthRetriever()

    if args.agent == 'e2e':
        agent = E2EAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'cot':
        agent = CoTAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'selfcst':
        agent = SelfConsistencyAgent(llm_config, result_path_root, retriever)

    return args, samples, agent


async def process_queries(queries, agent):
    await tqdm_asyncio.gather(*(agent.query(query) for query in queries))


async def main():
    args, samples, agent = init()
    
    # load embeddings
    # if args.retriever not in ['none', 'gth']:
    #     await prepare_emb(args, samples, llm_config)

    if args.debug:
        print(agent.generate_query(samples[0]))
        return

    start = args.start
    end = args.end
    batch = args.batch

    if not args.eval:
        for i in range(start, end, batch):
            left, right = i, min(i+batch, end)
            print(f"Run sample {left}-{right-1}")
            await process_queries(samples[left:right], agent)

            for j in range(left, right):
                # uid = samples[j]['uid']
                with open(os.path.join(args.result_path_root, f'{j}.json'), 'r') as file:
                    result = json.loads(file.read())
                pred = result['prediction'].strip('\n').split('Answer: ')[-1]
                if args.think and '<think>' in pred and '</think>' not in pred:
                    pred = ""
                result['prediction'] = pred
                if args.dev:
                    gold = samples[j]['qa']['answer']
                    result['gold'] = gold
                    # results.append({'uid': uid, 'pred': pred, 'gold': gold})            
                # else:
                    # results.append({'uid': uid, 'predicted_ans': pred, 'predicted_program': []})
                with open(os.path.join(args.result_path_root, f'{j}.json'), 'w') as file:
                    file.write(json.dumps(result, indent=2))

    # evaluate
    if args.dev:
        exact, f1 = 0.0, 0.0
        for i in range(end-start):
            with open(os.path.join(args.result_path_root, f'{i}.json'), 'r') as file:
                result = json.loads(file.read())
            pred, gold = result['prediction'], str(result['gold'])
            exact_acc, f1_acc = get_span_selection_metrics(pred, gold)
            exact += exact_acc
            f1 += f1_acc
        exact = exact / (end-start)
        f1 = f1 / (end-start)
        print(f'Exact Match: {exact*100:.2f}, F1: {f1*100:.2f}')

        if args.retriever != "none":
            text_pre, text_rec, text_ndcg, table_pre, table_rec, table_ndcg = 0, 0, 0, 0, 0, 0
            for i in range(end-start):
                with open(os.path.join(args.result_path_root, f'{i}.json'), 'r') as file:
                    result = json.loads(file.read())
                text_gth = list(dict.fromkeys(samples[i]['qa']['text_evidence']).keys())
                text_pred = list(dict.fromkeys(result['retrieved_text_ids']).keys())
                text_join = set(text_gth).intersection(set(text_pred))
                try:
                    text_pre += len(list(text_join)) / len(text_pred)
                except ZeroDivisionError:
                    text_pre += 1
                try:
                    text_rec += len(list(text_join)) / len(text_gth)
                except ZeroDivisionError:
                    text_rec += 1

                text_dcg, text_idcg = 0, 0
                for j, item in enumerate(text_pred):
                    if item in text_gth:
                        text_dcg += 1 / math.log2(j+2)
                for j in range(len(text_gth)):
                    if j == len(text_pred):
                        break
                    text_idcg += 1 / math.log2(j+2)
                try:
                    text_ndcg += text_dcg / text_idcg
                except ZeroDivisionError:
                    text_ndcg += 1

                if args.tabheader:
                    table_gth = list(dict.fromkeys(samples[i]['qa']['table_evidence']).keys())
                    table_pred = result['retrieved_table_ids']
                    cleaned_col_indices, cleaned_row_indices = [], []
                    for item in table_pred[0]:
                        for site in item[2]:
                            cleaned_col_indices.append((item[0], item[1], site))
                    for item in table_pred[1]:
                        for site in item[2]:
                            cleaned_row_indices.append((item[0], item[1], site))
                    hit = 0
                    for item in table_gth:
                        id, row, col = item.split('-')
                        id, row, col = int(id), int(row), int(col)
                        if (id, 'row', row) in cleaned_row_indices and (id, 'col', col) in cleaned_col_indices:
                            hit += 1
                    table_rec += hit / len(table_gth)
                else:
                    table_gth = list(dict.fromkeys(samples[i]['qa']['table_evidence']).keys())
                    table_pred = list(dict.fromkeys(result['retrieved_table_ids']).keys())
                    table_gth_dict = {key: j for j, key in enumerate(samples[i]['table_description'])}
                    table_gth_norm = [table_gth_dict[key] for key in table_gth]

                    table_join = set(table_gth_norm).intersection(set(table_pred))
                    try:
                        table_pre += len(list(table_join)) / len(table_pred)
                    except ZeroDivisionError:
                        table_pre += 1
                    try:
                        table_rec += len(list(table_join)) / len(table_gth_norm)
                    except ZeroDivisionError:
                        table_rec += 1

                    table_dcg, table_idcg = 0, 0
                    for j, item in enumerate(table_pred):
                        if item in table_gth_norm:
                            table_dcg += 1 / math.log2(j+2)
                    for j in range(len(table_gth_norm)):
                        if j == len(table_pred):
                            break
                        table_idcg += 1 / math.log2(j+2)
                    try:
                        table_ndcg += table_dcg / table_idcg
                    except ZeroDivisionError:
                        table_ndcg += 1
                    
            text_pre = text_pre / (end-start)
            text_rec = text_rec / (end-start)
            text_ndcg = text_ndcg / (end-start)
            print(f'Retrieved Texts Presicion: {text_pre*100:.2f}, Recall: {text_rec*100:.2f}, NDCG: {text_ndcg*100:.2f}')

            if args.tabheader:
                table_rec = table_rec / (end-start)
                print(f'Retrieved Tables Recall: {table_rec*100:.2f}')
            else:
                table_pre = table_pre / (end-start)
                table_rec = table_rec / (end-start)
                table_ndcg = table_ndcg / (end-start)
                print(f'Retrieved Tables Presicion: {table_pre*100:.2f}, Recall: {table_rec*100:.2f}, NDCG: {table_ndcg*100:.2f}')
    else:
        results = []
        for i in range(end-start):
            with open(os.path.join(args.result_path_root, f'{i}.json'), 'r') as file:
                result = json.loads(file.read())
            results.append({
                "uid": result["uid"], "predicted_ans": result["prediction"], "predicted_program": []
            })
        with open(os.path.join(args.result_path_root, f'test_predictions.json'), 'w') as file:
            file.write(json.dumps(results, indent=2))
        print('Predictions saved!')


if __name__ == '__main__':
    asyncio.run(main())
