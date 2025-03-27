import http.client
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def _embedding(model:str, api_key:str, input:str|list[str]):
   conn = http.client.HTTPSConnection("api2.aigcbest.top")
   payload = json.dumps({
      "model": model,
      "input": input
   })
   headers = {
      'Authorization': api_key,
      'Content-Type': 'application/json'
   }

   conn.request("POST", "/v1/embeddings", payload, headers)
   res = conn.getresponse()
   data = res.read()
   data=json.loads(data.decode('utf-8'))
   emb_vec=[]
   for vec in data['data']:
      emb_vec.append(np.array(vec['embedding']))
   
   return emb_vec

def similarity_cal(model:str, api_key:str, question:str, input:dict, top_k=5):
   input_value=list(input.values())
   input_value.append(question)
   input_emb=_embedding(model,api_key,input_value)
   n=len(input_emb)
   similarity=cosine_similarity([input_emb[n-1]],input_emb[0:n-1])
   similarity_dict={key:value for key, value in zip(input.keys(), similarity[0].tolist())}
   sorted_list = [{key: value} for key, value in sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)]

   if len(sorted_list)<top_k:
      return sorted_list
   else:
      return sorted_list[0:top_k]