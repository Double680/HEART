import os
import json
import argparse
from baselines.base_agent import *
from utils.evaluate import *
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import asyncio


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--agent', default='e2e')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--batch', default=100, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.dev:
        input_path = 'datasets/multihiertt/dev.json'
    else:
        input_path = 'datasets/multihiertt/test.json'
    with open(input_path, 'r') as file:
        samples = json.loads(file.read())
        for i in range(len(samples)):
            samples[i]['id'] = i

    result_path_root = f'./results/{args.agent}'
    if not os.path.exists(result_path_root):
        os.mkdir(result_path_root)

    if args.dev:
        result_path_root = os.path.join(result_path_root, 'dev')
    else:
        result_path_root = os.path.join(result_path_root, 'test')
    if not os.path.exists(result_path_root):
        os.mkdir(result_path_root)
    args.result_path_root = result_path_root

    load_dotenv()
    llm_config = {
        "llm_model": os.getenv('LLM_MODEL'),
        "emb_model": os.getenv('EMB_MODEL'),
        "api_key": os.getenv('API_KEY'),
        "base_url": os.getenv('BASE_URL'),
    }

    if args.agent == 'e2e':
        agent = E2EAgent(llm_config, result_path_root)
    elif args.agent == 'cot':
        agent = CoTAgent(llm_config, result_path_root)
    elif args.agent == 'selfcst':
        agent = SelfConsistencyAgent(llm_config, result_path_root)
    elif args.agent == 'o3-mini':
        llm_config['model'] = 'o3-mini-high'
        agent = E2EAgent(llm_config, result_path_root)
    
    return args, samples, agent


async def process_queries(queries, agent):
    await tqdm_asyncio.gather(*(agent.query(query) for query in queries))


async def main():
    args, samples, agent = init()
    
    if args.debug:
        print(agent.generate_query(samples[0]))
        return

    start = args.start
    end = len(samples) if args.end == -1 else args.end
    batch = args.batch
    results = []

    for i in range(start, end, batch):
        left, right = i, min(i+batch, end)
        await process_queries(samples[left:right], agent)

        for j in range(left, right):
            uid = samples[j]['uid']
            with open(os.path.join(args.result_path_root, f'{j}.json'), 'r') as file:
                result = json.loads(file.read())
            pred = result['prediction'].split('Answer: ')[-1]
            result['prediction'] = pred
            if args.dev:
                gold = samples[j]['qa']['answer']
                result['gold'] = gold
                results.append({'uid': uid, 'pred': pred, 'gold': gold})            
            else:
                results.append({'uid': uid, 'predicted_ans': pred, 'predicted_program': []})
            with open(os.path.join(args.result_path_root, f'{j}.json'), 'w') as file:
                file.write(json.dumps(result, indent=2))

    if args.dev:
        exact, f1 = 0.0, 0.0
        for i in range(end-start):
            pred, gold = results[i]['pred'], str(results[i]['gold'])
            exact_acc, f1_acc = get_span_selection_metrics(pred, gold)
            exact += exact_acc
            f1 += f1_acc
        exact = exact / (end-start)
        f1 = f1 / (end-start)
        print(f'Exact Match: {exact*100:.2f}, F1: {f1*100:.2f}')
    else:
        with open(os.path.join(args.result_path_root, 'test_predictions.json'), 'w') as file:
            file.write(json.dumps(results, indent=2))
        print('Predictions saved!')


if __name__ == '__main__':
    asyncio.run(main())
