import os
import json
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
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--batch', default=100, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--think', action='store_true')
    args = parser.parse_args()

    # load data
    if args.dev:
        input_path = './datasets/multihiertt/dev.json'
    else:
        input_path = './datasets/multihiertt/test.json'
    with open(input_path, 'r') as file:
        samples = json.loads(file.read())
        for i in range(len(samples)):
            samples[i]['id'] = i
    if args.end == -1:
        args.end = len(samples)

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
    result_path_agent = os.path.join(result_path_model, f'{args.agent}-{args.retriever}-{args.aug}')
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
            retriever = DensePassageRetriever(args.stored_emb_dir, args.aug, top_k=20, gpu=args.gpu)
        elif args.retriever == "gth":
            retriever = GroundTruthRetriever()

    if args.agent == 'e2e':
        agent = E2EAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'cot':
        agent = CoTAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'selfcst':
        agent = SelfConsistencyAgent(llm_config, result_path_root, retriever)

    return args, samples, llm_config, agent


async def process_queries(queries, agent):
    await tqdm_asyncio.gather(*(agent.query(query) for query in queries))


async def main():
    args, samples, llm_config, agent = init()
    
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
                try:
                    pred = result['response'].strip('\n').split('Answer: ')[-1].split('**Answer**: ')[-1].split('**Answer:** ')[-1].strip("*")
                except Exception:
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
            text_pre, text_rec, table_pre, table_rec = 0, 0, 0, 0
            for i in range(end-start):
                with open(os.path.join(args.result_path_root, f'{i}.json'), 'r') as file:
                    result = json.loads(file.read())
                text_gth, text_pred = list(set(samples[i]['qa']['text_evidence'])), list(set(result['retrieved_text_ids']))
                text_join = set(text_gth).intersection(set(text_pred))
                try:
                    text_pre += len(list(text_join)) / len(text_pred)
                except ZeroDivisionError:
                    text_pre += 1
                try:
                    text_rec += len(list(text_join)) / len(text_gth)
                except ZeroDivisionError:
                    text_rec += 1
                table_gth, table_pred = list(set(samples[i]['qa']['table_evidence'])), list(set(result['retrieved_table_ids']))
                table_gth_norm = []
                for i, key in enumerate(samples[i]['table_description']):
                    if key in table_gth:
                        table_gth_norm.append(i)
                table_join = set(table_gth_norm).intersection(set(table_pred))
                try:
                    table_pre += len(list(table_join)) / len(table_pred)
                except ZeroDivisionError:
                    table_pre += 1
                try:
                    table_rec += len(list(table_join)) / len(table_gth)
                except ZeroDivisionError:
                    table_rec += 1
            text_pre = text_pre / (end-start)
            text_rec = text_rec / (end-start)
            table_pre = table_pre / (end-start)
            table_rec = table_rec / (end-start)
            print(f'Retrieved Texts Presicion: {text_pre*100:.2f}, Recall: {text_rec*100:.2f}')
            print(f'Retrieved Tables Presicion: {table_pre*100:.2f}, Recall: {table_rec*100:.2f}')
    else:
        results = []
        for i in range(end-start):
            with open(os.path.join(args.result_path_root, f'{i}.json'), 'r') as file:
                result = json.loads(file.read())
            results.append({
                "uid": result["uid"], "predicted_ans": result["prediction"], "predicted_program": []
            })
        with open(os.path.join(args.result_path_root, 'test_predictions.json'), 'w') as file:
            file.write(json.dumps(results, indent=2))
        print('Predictions saved!')


if __name__ == '__main__':
    asyncio.run(main())
