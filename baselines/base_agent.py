from openai import AsyncOpenAI
from utils.api_query import *
from copy import copy
from collections import Counter
import time
import json
import os


def save_prediction(uid, pred, path):
    result = {
        'uid': uid,
        'prediction': pred
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


class BaseAgent:
    def __init__(self, config, result_path, cot=False):
        self.aclient = AsyncOpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        self.llm_model = config['llm_model']
        self.emb_model = config['emb_model']
        self.result_path = result_path

        self.cot = cot
        if self.cot:
            self.system_message = "According to the passage, please answer the question by generating the chain of thought. Please end up with 'Answer: XXX'. "
            self.message_template = "Passage: \n<PASSAGE>\nQuestion: <QUESTION>\nSolution: \n<SOLUTION>"
        else:
            self.system_message = "According to the passage, please answer the question directly with no other text."
            self.message_template = "Passage: \n<PASSAGE>\nQuestion: <QUESTION>\nAnswer: <ANSWER>"

    def _get_template_message(self, passage='', question='', answer='____'):
        template_message = self.message_template
        template_message = template_message.replace('<PASSAGE>', passage)
        template_message = template_message.replace('<QUESTION>', question)

        if self.cot:
            template_message = template_message.replace('<SOLUTION>', answer)
        else:
            template_message = template_message.replace('<ANSWER>', answer)

        return template_message
    
    def generate_examples(self):
        examples = "Here are some examples of answer forms: \n"
        solution_prefix = "Let's think step by step. XXX\nAnswer: " if self.cot else ""
        examples += 'Example 1\n' + self._get_template_message('XXX', 'In which year XXX?', f'{solution_prefix}2025 2026\n\n')
        examples += 'Example 2\n' + self._get_template_message('XXX', 'Does the score XXX?', f'{solution_prefix}yes\n\n')
        examples += 'Example 3\n' + self._get_template_message('XXX', 'Who wrote XXX?', f'{solution_prefix}jane smith\n\n')
        examples += 'Example 4\n' + self._get_template_message('XXX', 'What is the rate XXX?', f'{solution_prefix}0.5386\n\n')
        return examples
    
    def generate_query_message(self, sample):
        processed_sample = process_raw_sample(sample)
        query_message = self.generate_examples()

        if self.cot:
            query_message += "Please answer the question according to the passage content. Remember to end up with 'Answer: <ANSWER>'\n"
        else:
            if self.llm_model != "o3-mini-high":
                query_message += "Please answer the question according to the passage content. \n"
            else: 
                query_message += "Please generate the answer to the question according to the passage content. Do not generate other texts, such as the intermediate thinking. \n"
        query_message += self._get_template_message(processed_sample['document'], processed_sample['question'])
        return query_message

    def _preprocess_query(self, sample):
        id, uid = sample['id'], sample['uid']
        path = os.path.join(self.result_path, f'{id}.json')
        query_message = self.generate_query_message(sample)
        messages=[
            {'role': 'system', 'content': self.system_message}, 
            {'role': 'user', 'content': query_message}
        ]
        if self.llm_model == "o3-mini-high":
            messages = messages[1:]
        return uid, messages, path


class E2EAgent(BaseAgent):
    def __init__(self, config, result_path):
        super().__init__(self, config, result_path)
        
    async def query(self, sample):
        uid, messages, path = self._preprocess_query(sample)
        if os.path.exists(path):
            return
        while True:
            try:
                completion = await llm_query(self.aclient, self.llm_model, messages, max_tokens=100)
                response = completion.choices[0].message.content
                break
            except Exception:
                print('API Calling failed, retry in 5 seconds...')
                time.sleep(5)
        
        save_prediction(uid, response, path)
    

class CoTAgent(BaseAgent):
    def __init__(self, config, result_path):
        super().__init__(self, config, result_path)

    async def query(self, sample):
        uid, messages, path = self._preprocess_query(sample)
        if os.path.exists(path):
            return
        while True:
            try:
                completion = await llm_query(self.aclient, self.llm_model, messages)
                response = completion.choices[0].message.content
                break
            except Exception:
                print('API Calling failed, retry in 5 seconds...')
                time.sleep(5)
        
        save_prediction(uid, response, path)
        

class SelfConsistencyAgent(BaseAgent):
    def __init__(self, config, result_path):
        super().__init__(self, config, result_path, cot=True)

    async def query(self, sample):
        uid, messages, path = self._preprocess_query(sample)
        if os.path.exists(path):
            return
        while True:
            try:
                completion = await llm_query(self.aclient, self.llm_model, messages, n=5)
                responses = [item.message.content.split('Answer: ')[-1] for item in completion.choices]
                counter = Counter(responses)
                response = counter.most_common(1)[0][0]
                break
            except Exception:
                print('API Calling failed, retry in 5 seconds...')
                time.sleep(5)

        save_prediction(uid, response, path)
