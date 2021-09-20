#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import glob
import ast
import json
import numpy as np
import os
from pathlib import Path


path_iclr_train = r'../0517_split_2/train/' # use your path
path_iclr_test = r'../0517_split_2/test/' # use your path
path_iclr_dev = r'../0517_split_2/dev/' # use your path

label_name_classification = "coarse"


# In[39]:


# keys are label types here
# label_keys = ["coarse","fine","asp","pol"]
label_keys = ["coarse"]


# ## Dev Data

# In[44]:


all_review_files_dev = glob.glob(path_iclr_dev + "*")

data_li = []

for filename in all_review_files_dev:
    
  with open(filename, 'r') as f:
    data_json =  json.loads(f.read())
  rev_df = pd.DataFrame(data_json["review"])
  rev_label_df = pd.DataFrame(data_json["reviewlabels"])
  df = pd.concat([rev_df,rev_label_df],axis=1)
  df = df.loc[:,~df.columns.duplicated()]
  data_li.append(df)


# In[45]:


for label_key in label_keys:
    json_li = []
    for data in data_li:
        json_li.append({"review_id":data["text_id"][0],"sentences":[sentence for sentence in data["sentence"]],"labels":[req_label for req_label in [label[label_key] for label in data["labels"]]]})
    Path("./data/Discourse").mkdir(parents=True, exist_ok=True)
    with open(f'./data/Discourse/dev_{label_key}.jsonl', 'w') as outfile:
        for entry in json_li:
            json.dump(entry, outfile)
            outfile.write('\n')


# ## Training Data

# In[34]:


all_review_files_train = glob.glob(path_iclr_train + "*")

data_li = []

for filename in all_review_files_train:
    
  with open(filename, 'r') as f:
    data_json =  json.loads(f.read())
  rev_df = pd.DataFrame(data_json["review"])
  rev_label_df = pd.DataFrame(data_json["reviewlabels"])
  df = pd.concat([rev_df,rev_label_df],axis=1)
  df = df.loc[:,~df.columns.duplicated()]
  data_li.append(df)


# In[43]:



for label_key in label_keys:
    json_li = []
    for data in data_li:
        json_li.append({"review_id":data["text_id"][0],"sentences":[sentence for sentence in data["sentence"]],"labels":[req_label for req_label in [label[label_key] for label in data["labels"]]]})
    Path("./data/Discourse").mkdir(parents=True, exist_ok=True)
    with open(f'./data/Discourse/train_{label_key}.jsonl', 'w') as outfile:
        for entry in json_li:
            json.dump(entry, outfile)
            outfile.write('\n')


# ## Testing Data

# In[46]:


all_review_files_test = glob.glob(path_iclr_test + "*")

data_li = []

for filename in all_review_files_test:
    
  with open(filename, 'r') as f:
    data_json =  json.loads(f.read())
  rev_df = pd.DataFrame(data_json["review"])
  rev_label_df = pd.DataFrame(data_json["reviewlabels"])
  df = pd.concat([rev_df,rev_label_df],axis=1)
  df = df.loc[:,~df.columns.duplicated()]
  data_li.append(df)


# In[47]:


for label_key in label_keys:
    json_li = []
    for data in data_li:
        json_li.append({"review_id":data["text_id"][0],"sentences":[sentence for sentence in data["sentence"]],"labels":[req_label for req_label in [label[label_key] for label in data["labels"]]]})
    Path("./data/Discourse").mkdir(parents=True, exist_ok=True)
    with open(f'./data/Discourse/test_{label_key}.jsonl', 'w') as outfile:
        for entry in json_li:
            json.dump(entry, outfile)
            outfile.write('\n')


# In[ ]:




