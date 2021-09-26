
# coding: utf-8

# In[5]:

import pandas as pd
import glob
import ast
import json
import numpy as np
import os
from pathlib import Path


path_iclr_train = r'../../Discourse_Final_Dataset/train/' # use your path
path_iclr_test = r'../../Discourse_Final_Dataset/test/' # use your path
path_iclr_dev = r'../../Discourse_Final_Dataset/dev/' # use your path

label_name_classification = "coarse"


# In[6]:

# keys are label types here
# label_keys = ["coarse","fine","asp","pol"]
label_keys = ["coarse"]


# ## Dev Data

# In[8]:

all_review_files_dev = glob.glob(path_iclr_dev + "*")

data_li = []

for filename in all_review_files_dev:
    
  with open(filename, 'r') as f:
    data_json =  json.loads(f.read())
  rev_df = pd.DataFrame(data_json["review_sentences"])
  data_li.append(rev_df)


# In[10]:

for label_key in label_keys:
    json_li = []
    for data in data_li:
        json_li.append({"review_id":data["review_id"][0],"sentences":[sentence for sentence in data["text"]],"labels":[req_label for req_label in data[label_key]]})
    Path("../data/Discourse").mkdir(parents=True, exist_ok=True)
    with open(f'../data/Discourse/dev_{label_key}.jsonl', 'w') as outfile:
        for entry in json_li:
            json.dump(entry, outfile)
            outfile.write('\n')


# ## Training Data

# In[11]:

all_review_files_train = glob.glob(path_iclr_train + "*")

data_li = []

for filename in all_review_files_train:
    
  with open(filename, 'r') as f:
    data_json =  json.loads(f.read())
  rev_df = pd.DataFrame(data_json["review_sentences"])
  data_li.append(rev_df)


# In[12]:


for label_key in label_keys:
    json_li = []
    for data in data_li:
        json_li.append({"review_id":data["review_id"][0],"sentences":[sentence for sentence in data["text"]],"labels":[req_label for req_label in data[label_key]]})
    Path("../data/Discourse").mkdir(parents=True, exist_ok=True)
    with open(f'../data/Discourse/train_{label_key}.jsonl', 'w') as outfile:
        for entry in json_li:
            json.dump(entry, outfile)
            outfile.write('\n')


# ## Testing Data

# In[14]:

all_review_files_test = glob.glob(path_iclr_test + "*")

data_li = []

for filename in all_review_files_test:
    with open(filename, 'r') as f:
        data_json =  json.loads(f.read())
    rev_df = pd.DataFrame(data_json["review_sentences"])
    data_li.append(rev_df)


# In[15]:

for label_key in label_keys:
    json_li = []
    for data in data_li:
        json_li.append({"review_id":data["review_id"][0],"sentences":[sentence for sentence in data["text"]],"labels":[req_label for req_label in data[label_key]]})
    Path("../data/Discourse").mkdir(parents=True, exist_ok=True)
    with open(f'../data/Discourse/test_{label_key}.jsonl', 'w') as outfile:
        for entry in json_li:
            json.dump(entry, outfile)
            outfile.write('\n')

