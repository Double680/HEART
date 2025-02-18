import os
import json
import argparse
from baselines.base_agent import *
from utils.evaluate import *
import copy
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio

async def process_queries(queries, agent):
    preds = await tqdm_asyncio.gather(*(agent.query(query) for query in queries))
    return preds

async def main(args):
    if args.dev:
        input_path = 'datasets/multihiertt/dev.json'
    else:
        input_path = 'datasets/multihiertt/test.json'
    with open(input_path, 'r') as file:
        data = json.loads(file.read())

    llm_config = {
        "model": args.model,
        "api_key": "sk-HDB7FhzGBbbDKUEKyx5NWzKjTpBfmR1FNzK2v4rrPXfSaxCT",
        "base_url": "https://api2.aigcbest.top/v1",
    }

    if args.agent == 'e2e':
        agent = E2EAgent(llm_config)
    elif args.agent == 'cot':
        agent = CoTAgent(llm_config)
    elif args.agent == 'selfcst':
        agent = SelfConsistencyAgent(llm_config)

    samples = []
    for id in tqdm(range(len(data))):
        case = data[id]
        paragraphs = copy.copy(case['paragraphs'])
        table_cnt = 0
        for i in range(len(paragraphs)):
            if paragraphs[i] == f'## Table {table_cnt} ##':
                paragraphs[i] = case['tables'][table_cnt]
                table_cnt += 1
        assert table_cnt == len(case['tables'])
        document = '\n'.join(paragraphs)
        if args.dev:
            samples.append({
                'uid': case['uid'], 'document': document, 'question': case['qa']['question'], 'answer': case['qa']['answer']
            })     
        else:
            samples.append({
                'uid': case['uid'], 'document': document, 'question': case['qa']['question']
            })   

    if args.debug:
        print(agent.generate_query(samples[0]))
        return

    start = args.start
    end = len(samples) if args.end == -1 else args.end
    results = []

    result_path_root = f'./results/{args.agent}'
    if not os.path.exists(result_path_root):
        os.mkdir(result_path_root)

    result_path_root = os.path.join(result_path_root, args.model)
    if not os.path.exists(result_path_root):
        os.mkdir(result_path_root)

    if args.dev:
        result_path_root = os.path.join(result_path_root, 'dev')
    else:
        result_path_root = os.path.join(result_path_root, 'test')
    if not os.path.exists(result_path_root):
        os.mkdir(result_path_root)

    if args.eval:
        for i in range(start, end):
            uid = samples[i]['uid']
            with open(os.path.join(result_path_root, f'{i}.txt'), 'r') as file:
                pred = file.read()
            pred = pred.split('Gold:')[0]
            pred = pred.split('Answer: ')[-1].split('Prediction:\n')[-1].strip('\n')
            if args.dev:
                gold = samples[i]['answer']
                results.append({'uid': uid, 'pred': pred, 'gold': gold})            
            else:
                results.append({'uid': uid, 'predicted_ans': pred, 'predicted_program': []})
    else:
        preds = []
        batch = 100
        for i in range(start, end, batch):
            preds += await process_queries(samples[i:min(i+batch, end)], agent)

        for i in range(start, end):
            uid = samples[i]['uid']
            pred = preds[i-start]
            
            with open(os.path.join(result_path_root, f'{i}.txt'), 'w') as file:
                file.write(f"UID:\n{uid}\n")
                file.write(f"Prediction:\n{pred}\n")
                pred = pred.split('Answer: ')[-1]
                if args.dev:
                    gold = samples[i]['answer']
                    file.write(f"Gold:\n{gold}")
                    results.append({'uid': uid, 'pred': pred, 'gold': gold})            
                else:
                    results.append({'uid': uid, 'predicted_ans': pred, 'predicted_program': []})

    if args.dev:
        exact, f1 = 0.0, 0.0
        for i in range(end-start):
            pred, gold = results[i]['pred'], str(results[i]['gold'])
            exact_acc, f1_acc = get_span_selection_metrics(pred, gold)
            exact += exact_acc
            f1 += f1_acc
        exact = exact / (end-start)
        f1 = f1 / (end-start)
        print(exact, f1)
    else:
        with open(os.path.join(result_path_root, 'test_predictions.json'), 'a') as file:
            file.write(json.dumps(results, indent=2))
        print('Predictions saved!')

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--model', default='gpt-4o-mini-2024-07-18')
parser.add_argument('--agent', default='e2e')
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--end', default=-1, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

asyncio.run(main(args))
