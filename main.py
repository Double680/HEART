import asyncio
from tqdm.asyncio import tqdm_asyncio
from modules.init import *
from modules.evaluate import *


def process_output(args, left, right):
    save_root = args.save_root_setting
    for j in range(left, right):
        with open(os.path.join(save_root, f'{j}.json'), 'r') as file:
            result = json.loads(file.read())
        try:
            pred = result['response'].strip('\n').split('<answer>')[-1].split('</answer>')[0].strip('*').strip()
        except Exception:
            pred = ""
        result['prediction'] = pred
        if args.dev:
            gold = args.samples[j]['qa']['answer']
            result['gold'] = gold
        with open(os.path.join(save_root, f'{j}.json'), 'w') as file:
            file.write(json.dumps(result, indent=2))


async def run_sample(args):
    agent = args.agent
    start = args.start
    end = args.end
    batch = args.batch
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def query_with_limit(sample):
        async with semaphore:
            await agent.query(sample)

    for i in range(start, end, batch):
        left, right = i, min(i+batch, end)
        print(f"Run sample {left}-{right-1}")
        await tqdm_asyncio.gather(*(query_with_limit(sample) for sample in args.samples[left:right]))

        process_output(args, left, right)


async def main():
    args = init()
    if not args.eval:
        await run_sample(args)
    evaluate(args)


if __name__ == "__main__":
    asyncio.run(main())
