from vllm import LLM

class Reranker:
    def __init__(self, model_path):
        self.reranker = LLM(
            model=model_path,
            task="score",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
            gpu_memory_utilization=0.4
        )
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        self.instruction = "Judge whether the table row/column is relevant to answer the question. "
    
    def query_format(self, query):
        return self.query_template.format(
            prefix=self.prefix, 
            instruction=self.instruction, 
            query=query
        )
    
    def doc_format(self, doc):
        return self.document_template.format(
            doc=doc, 
            suffix=self.suffix
        )


import requests
import json
import time


def call_reranker_vllm_online(query, docs, model_name="models/Qwen3-Reranker-0.6B", host="localhost", port=8000):
    url = f"http://{host}:{port}/rerank"

    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"

    instruction = "Judge whether the table row/column is relevant to answer the question. "
    
    # 准备请求数据
    data = {
        "model": model_name,
        "query": query_template.format(
            prefix=prefix, 
            instruction=instruction, 
            query=query
        ),
        "documents": [
            document_template.format(
                doc=doc,
                suffix=suffix
            )
            for doc in docs
        ]
    }
    
    while True:
        try:
            # 发送 POST 请求
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result  # 返回生成的文本
            else:
                print("Bad response: ", response.status_code)
                
        except Exception as e:
            time.sleep(2)
            print(f"Error: {str(e)}")