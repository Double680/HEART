from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from augment.retriever import Retriever
from transformers import AutoModelForCausalLM
import torch

train_data_path = 'datasets/multihiertt/train.json'
retriever_model_path = 'models/Qwen3-Embedding-0.6B'
augment_model_path = 'models/Qwen3-1.7B'
save_model_path = 'models/HybTQA-augment'

dataset = load_dataset('json', data_files=train_data_path)

def make_conversation(example):
    template = "Modify the given question precisely by adding more details within <query> </query> tags. \nQuestion: <QUESTION> /no_think"
    prompt = {
        "prompt": [{
            "role": "user", 
            "content": template.replace("<QUESTION>", example["qa"]["question"])
        }]
    }
    return prompt

dataset = dataset.map(make_conversation)
dataset = dataset.remove_columns(["uid", "tables"])
dataset = dataset["train"]

retriever = Retriever(retriever_model_path)

def reward_value(value):
    return 4 * value
    
def format_reward(question):
    reward = -2
    if "<query>" in question and "</query>" in question:
        reward += 1
    if len(question.split("<query>")[-1].split("</query>")[0]) > 0:
        reward += 1
    return reward

def reward_func(completions, **kwargs):
    questions = [completion[0]["content"].split('</think>')[-1].strip('\n') for completion in completions]
    format_rewards = [format_reward(question) for question in questions]
    questions = [
        question.split("<query>")[-1].split("</query>")[0].strip('\n') for question in questions
    ]
    text_docs = kwargs["paragraphs"]
    table_docs = kwargs["table_description"]
    text_evids = [kwargs["qa"][id]["text_evidence"] for id in range(len(kwargs["qa"]))]
    table_evids = [kwargs["qa"][id]["table_evidence"] for id in range(len(kwargs["qa"]))]
    text_scores = [
        retriever.eval(retriever.retrieve(question, text_doc), text_evid)[2] 
        for question, text_doc, text_evid in zip(questions, text_docs, text_evids)
    ]
    table_scores = [
        retriever.eval(retriever.retrieve(question, table_doc), table_evid)[2] 
        for question, table_doc, table_evid in zip(questions, table_docs, table_evids)
    ]
    retrieve_rewards = [
        reward_value(text_score) + reward_value(table_score)
        for text_score, table_score in zip(text_scores, table_scores)
    ]
    rewards = [fr + rr for fr, rr in zip(format_rewards, retrieve_rewards)]
    return rewards

training_args = GRPOConfig(
    output_dir="HybTQA-test",
    learning_rate=5e-6,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    max_completion_length=256,
    num_generations=8,
    max_prompt_length=128,
    logging_steps=5,
    save_steps=2000,
    report_to=None
)

model = AutoModelForCausalLM.from_pretrained(
    "/data/private/models/Qwen3-1.7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(training_args.output_dir)