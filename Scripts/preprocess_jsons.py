import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list}  # removed quotes around variable
    )
    return r.json()["embeddings"]

jsons = os.listdir("jsons")  # list all the jsons
print(jsons)
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")
    embeddings = []

    # loop over chunks in this file
    for c in content['chunks']:
        emb = create_embedding([c['text']])[0]  # get single embedding
        embeddings.append(emb)

    # assign embeddings to chunks
    for i, chunk in enumerate(content['chunks']):
        print(chunk)
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
        # if(i==3): # Read 3 chunks 
        #     break

# make DataFrame
df = pd.DataFrame.from_records(my_dicts)
# print(df)
# save this dataframe
joblib.dump(df, 'embeddings.joblib')
