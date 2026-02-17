import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import requests
import joblib


def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input": text_list
    })
    embedding=r.json()["embeddings"]
    return embedding

def inference(prompt):
    r=requests.post("http://localhost:11434/api/generate",json={
        "model":"llama3.2",
        "prompt": prompt,
        "stream":False
    })
    response=r.json()
    print(response)
    return response
    



df=joblib.load("embeddings.joblib")

incoming_query=input("Ask your question: ")
question_embedding=create_embedding([incoming_query])[0]



similarity=cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
#print(similarity)
top_result=10
max_indx=similarity.argsort()[::-1][0:top_result]
#print(max_indx)
new_df=df.loc[max_indx]
#print(new_df["text"])

prompt=f'''Here are viodeo chunks conatining start time in seconds and end time in seconds along with text at that time:
{new_df[['start','end','text']].to_json(orient='records')}

----------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in human way (dont mention the above format its just for you) where and how much content is taught where(in which video at what timestamps)and guide the user to go to that particular video . If users ask for unrelated question, reply them you can only ask questions related to the course
'''
with open("prompt.txt","w") as f:
    f.write(prompt)

response=inference(prompt)['response']
print(response)
'''for index, item in new_df.iterrows():
    print(index,item["text"],item['start'],item['end'])'''

with open("response.txt","w") as f:
    f.write(response)
