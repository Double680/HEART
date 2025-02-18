from openai import AsyncOpenAI
import time

class E2EAgent:
    def __init__(self, config):
        self.aclient = AsyncOpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        self.model=config['model']
        self.system_message = "According to the passage, please answer the question directly with no other text."
        self.message_template = "Passage: \n<PASSAGE>\nQuestion: <QUESTION>\nAnswer: <ANSWER>"

    def _get_template_message(self, passage='', question='', answer='____'):
        template_message = self.message_template
        template_message = template_message.replace('<PASSAGE>', passage)
        template_message = template_message.replace('<QUESTION>', question)
        template_message = template_message.replace('<ANSWER>', answer)
        return template_message
        
    def generate_examples(self):
        examples = "Here are some examples of answer forms: \n"
        examples += 'Example 1\n' + self._get_template_message('XXX', 'In which year XXX?', '2025 2026\n\n')
        examples += 'Example 2\n' + self._get_template_message('XXX', 'Does the score XXX?', 'yes\n\n')
        examples += 'Example 3\n' + self._get_template_message('XXX', 'Who wrote XXX?', 'jane smith\n\n')
        examples += 'Example 4\n' + self._get_template_message('XXX', 'What is the rate XXX?', '0.5386\n\n')
        return examples

    def generate_query(self, sample):
        query_message = self.generate_examples()
        query_message += "Please answer the question according to the passage content. \n"
        query_message += self._get_template_message(sample['document'], sample['question'])
        return query_message

    async def query(self, sample):
        query_message = self.generate_query(sample)
        while True:
            try:
                if self.model not in ["o3-mini-high"]:
                    completion = await self.aclient.chat.completions.create(
                        model=self.model,
                        temperature=0.1,
                        max_tokens=100,
                        messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': query_message}]
                    )
                else:
                    completion = await self.aclient.chat.completions.create(
                        model=self.model,
                        max_tokens=100,
                        messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': query_message}]
                    )
                response = completion.choices[0].message.content
                break
            except Exception:
                print('API Calling failed, retry in 5 seconds...')
                time.sleep(5)
        return response
    
class CoTAgent:
    def __init__(self, config):
        self.aclient = AsyncOpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        self.model=config['model']
        self.system_message = "According to the passage, please answer the question by generating the chain of thought. Please end up with 'Answer: XXX'. "
        self.message_template = "Passage: \n<PASSAGE>\nQuestion: <QUESTION>\nSolution: \n<SOLUTION>"

    def _get_template_message(self, passage='', question='', answer='____'):
        template_message = self.message_template
        template_message = template_message.replace('<PASSAGE>', passage)
        template_message = template_message.replace('<QUESTION>', question)
        template_message = template_message.replace('<SOLUTION>', answer)
        return template_message
        
    def generate_examples(self):
        examples = "Here are some examples of answer forms: \n"
        examples += 'Example 1\n' + self._get_template_message('XXX', 'In which year XXX?', 'Let\'s think step by step. XXX\nAnswer: 2025 2026\n\n')
        examples += 'Example 2\n' + self._get_template_message('XXX', 'Does the score XXX?', 'Let\'s think step by step. XXX\nAnswer: yes\n\n')
        examples += 'Example 3\n' + self._get_template_message('XXX', 'Who wrote XXX?', 'Let\'s think step by step. XXX\nAnswer: jane smith\n\n')
        examples += 'Example 4\n' + self._get_template_message('XXX', 'What is the rate XXX?', 'Let\'s think step by step. XXX\nAnswer: 0.5386\n\n')
        return examples

    def generate_query(self, sample):
        query_message = self.generate_examples()
        query_message += "Please answer the question according to the passage content. Remember to end up with 'Answer: <ANSWER>'\n"
        query_message += self._get_template_message(sample['document'], sample['question'])
        return query_message

    async def query(self, sample):
        query_message = self.generate_query(sample)
        while True:
            try:
                completion = await self.aclient.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    max_tokens=1000,
                    messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': query_message}]
                )
                response = completion.choices[0].message.content
                break
            except Exception:
                print('API Calling failed, retry in 5 seconds...')
                time.sleep(5)
        return response
    
class SelfConsistencyAgent:
    def __init__(self, config):
        self.aclient = AsyncOpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        self.model=config['model']
        self.system_message = "According to the passage, please answer the question by generating the chain of thought. Please end up with 'Answer: XXX'. "
        self.message_template = "Passage: \n<PASSAGE>\nQuestion: <QUESTION>\nSolution: \n<SOLUTION>"

    def _get_template_message(self, passage='', question='', answer='____'):
        template_message = self.message_template
        template_message = template_message.replace('<PASSAGE>', passage)
        template_message = template_message.replace('<QUESTION>', question)
        template_message = template_message.replace('<SOLUTION>', answer)
        return template_message
        
    def generate_examples(self):
        examples = "Here are some examples of answer forms: \n"
        examples += 'Example 1\n' + self._get_template_message('XXX', 'In which year XXX?', 'Let\'s think step by step. XXX\nAnswer: 2025 2026\n\n')
        examples += 'Example 2\n' + self._get_template_message('XXX', 'Does the score XXX?', 'Let\'s think step by step. XXX\nAnswer: yes\n\n')
        examples += 'Example 3\n' + self._get_template_message('XXX', 'Who wrote XXX?', 'Let\'s think step by step. XXX\nAnswer: jane smith\n\n')
        examples += 'Example 4\n' + self._get_template_message('XXX', 'What is the rate XXX?', 'Let\'s think step by step. XXX\nAnswer: 0.5386\n\n')
        return examples

    def generate_query(self, sample):
        query_message = self.generate_examples()
        query_message += "Please answer the question according to the passage content. Remember to end up with 'Answer: <ANSWER>'\n"
        query_message += self._get_template_message(sample['document'], sample['question'])
        return query_message

    async def query(self, sample):
        query_message = self.generate_query(sample)
        while True:
            try:
                completion = await self.aclient.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    max_tokens=1000, 
                    messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': query_message}],
                    n=5,
                )
                responses = []
                for item in completion.choices:
                    responses.append(item.message.content.split('Answer: ')[-1])
                
                from collections import Counter
                counter = Counter(responses)
                response = counter.most_common(1)[0][0]
                break
            except Exception:
                print('API Calling failed, retry in 5 seconds...')
                time.sleep(5)
        return response