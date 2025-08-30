from openai import AsyncOpenAI
from utils.api_query import *
from copy import copy
import time
import json
import os


def save_prediction(uid, prompt, reasoning_content, response, path, text_inds, table_inds):
    result = {
        'uid': uid,
        'prompt': prompt,
        'reasoning_content': reasoning_content,
        'response': response, 
        'retrieved_text_ids': text_inds,
        'retrieved_table_ids': table_inds
    }
    with open(path, 'w') as file:
        file.write(json.dumps(result, indent=2))


def process_raw_sample(sample):
    paragraphs = copy(sample['paragraphs'])
    table_cnt = 0
    for i in range(len(paragraphs)):
        if paragraphs[i] == f'## Table {table_cnt} ##':
            paragraphs[i] = sample['tables'][table_cnt]
            table_cnt += 1
    assert table_cnt == len(sample['tables'])
    document = '\n'.join(paragraphs)
    processed_sample = {
        'uid': sample['uid'],
        'document': document,
        'question': sample['qa']['question'],
    }
    return processed_sample


class ThinkAgent:

    def __init__(self, config, result_path, retriever):

        self.aclient = AsyncOpenAI(base_url=config['base_url'], api_key=config['api_key'])
        self.llm_model = config['llm_model']
        self.result_path = result_path
        self.retriever = retriever
        self.message_template = "Answer the given question according to the given document. Put the direct answer within <answer></answer> tags. <document><DOCUMENT></document><question><QUESTION></question>"


    def preprocess(self, sample):

        id, uid = sample['id'], sample['uid']
        save_result_path = os.path.join(self.result_path, f'{id}.json')

        text_inds, table_inds = None, None
        if self.retriever != None:
            sample["paragraphs"], sample["tables"], text_inds, table_inds = self.retriever.retrieve(sample)

        processed_sample = process_raw_sample(sample)
        query_message = self.message_template
        document, question = processed_sample['document'], processed_sample['question']
        prompt = query_message.replace('<DOCUMENT>', document).replace('<QUESTION>', question)

        return uid, prompt, save_result_path, text_inds, table_inds


    async def query(self, sample):
        uid, prompt, path, text_inds, table_inds = self.preprocess(sample)
        if os.path.exists(path):
            return
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_content = None
        response = None

        cnt = 0
        while True:
            try:
                completion = await llm_query(self.aclient, self.llm_model, messages)
                try:
                    reasoning_content = completion.choices[0].message.reasoning_content
                except Exception:
                    reasoning_content = None
                response = completion.choices[0].message.content
                if reasoning_content is None:
                    if '<think>' in response:
                        reasoning_content = response.split('<think>')[1].split('</think>')[0]
                        response = response.split('</think>')[-1]
                break
            except Exception:
                cnt += 1
                if cnt >= 5:
                    break
                print('API Calling failed, retry in 3 seconds...')
                time.sleep(3)
        
        save_prediction(uid, prompt, reasoning_content, response, path, text_inds, table_inds)
    