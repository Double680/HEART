from openai import AsyncOpenAI


async def llm_query(aclient: AsyncOpenAI, model, messages, temperature=0.1, max_completion_tokens=2048, n=1):
    completion = await aclient.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        n=n
    )
    return completion


async def emb_query(aclient: AsyncOpenAI, model, inputs):
    embeddings = await aclient.embeddings.create(
        model=model,
        input=inputs 
    )
    return embeddings