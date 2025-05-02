import os
import json
import argparse
from baselines.base_agent import *
from baselines.retriever import *
from utils.evaluate import *
from utils.embedding import *
from utils.util import *
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import asyncio


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--agent', default='e2e')
    parser.add_argument('--retriever', default='none')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--batch', default=100, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
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

    result_path_dir = f'./results'
    result_path_agent = os.path.join(result_path_dir, args.agent)
    if args.dev:
        result_path_root = os.path.join(result_path_agent, 'dev')
    else:
        result_path_root = os.path.join(result_path_agent, 'test')

    ensure_dirs(result_path_dir, result_path_agent, result_path_root)
    args.result_path_root = result_path_root

    # load models
    load_dotenv()
    llm_config = {
        "llm_model": os.getenv('LLM_MODEL'),
        "emb_model": os.getenv('EMB_MODEL'),
        "api_key": os.getenv('API_KEY'),
        "base_url": os.getenv('BASE_URL'),
    }

    retriever = None
    if args.retriever != "none":
        stored_emb_dir_root = './embeddings'
        if args.dev:
            data_type = "dev"
        else:
            data_type = "test"
        stored_emb_dir = os.path.join(stored_emb_dir_root, data_type)
        args.stored_emb_dir = stored_emb_dir

        if args.retriever == "dpr":
            retriever = DensePassageRetriever(args.stored_emb_dir, gpu=args.gpu)

    if args.agent == 'e2e':
        agent = E2EAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'cot':
        agent = CoTAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'selfcst':
        agent = SelfConsistencyAgent(llm_config, result_path_root, retriever)
    elif args.agent == 'o3-mini':
        llm_config['llm_model'] = 'o3-mini-high'
        agent = E2EAgent(llm_config, result_path_root, retriever)

    return args, samples, llm_config, agent


async def process_queries(queries, agent):
    await tqdm_asyncio.gather(*(agent.query(query) for query in queries))


async def main():
    args, samples, llm_config, agent = init()
    
    # load embeddings
    if args.retriever != 'none':
        await prepare_emb(args, samples, llm_config)

    if args.debug:
        print(agent.generate_query(samples[0]))
        return

    start = args.start
    end = args.end
    batch = args.batch
    results = []

    for i in range(start, end, batch):
        left, right = i, min(i+batch, end)
        await process_queries(samples[left:right], agent)

        for j in range(left, right):
            uid = samples[j]['uid']
            with open(os.path.join(args.result_path_root, f'{j}.json'), 'r') as file:
                result = json.loads(file.read())
            pred = result['prediction'].strip('\n').split('Answer: ')[-1]
            result['prediction'] = pred
            if args.dev:
                gold = samples[j]['qa']['answer']
                result['gold'] = gold
                results.append({'uid': uid, 'pred': pred, 'gold': gold})            
            else:
                results.append({'uid': uid, 'predicted_ans': pred, 'predicted_program': []})
            with open(os.path.join(args.result_path_root, f'{j}.json'), 'w') as file:
                file.write(json.dumps(result, indent=2))

    # evaluate
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
