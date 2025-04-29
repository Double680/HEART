from openai import AsyncOpenAI

async def llm_query(aclient: AsyncOpenAI, model, messages, temperature=0.1, max_tokens=1000, n=1):
    completion = await aclient.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n
    )
    return completion