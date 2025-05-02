from openai import AsyncOpenAI

class DensePassageRetriever:
    def __init__(self, config):
        self.aclient = AsyncOpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        self.emb_model = config['emb_model']