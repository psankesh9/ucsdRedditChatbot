#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 


# In[4]:


import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset


# In[5]:


# ! pip install transformers


# In[6]:


from transformers import pipeline, set_seed, AutoTokenizer, GPT2Tokenizer, GPT2Model


# In[7]:


# from google.colab import drive

# drive.mount('/content/drive')


# In[8]:


# ! pip install opendatasets


# In[9]:


# import opendatasets as od


# In[10]:


# od.download(
#     "https://www.kaggle.com/datasets/luketaylor12345/ucsd-subreddit-dataset")


# In[11]:


file ='comments_clean.csv'


# In[12]:


comments_clean_df = pd.read_csv(file)


# In[13]:


file_subs ='subs.csv'


# In[14]:


subs_df = pd.read_csv(file_subs)


# In[15]:


comments_clean_df.head().iloc[0:1]


# In[16]:


# https://www.geeksforgeeks.org/how-to-import-kaggle-datasets-directly-into-google-colab/


# In[17]:


# !pip install datasets


# In[18]:


from transformers import pipeline, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, \
Trainer 
from datasets import Dataset


# In[19]:


# limit to shorter strings
#comments_train[comments_train['body'].apply(len) < 100].head(2)


# In[20]:


comments_clean_df.head()


# In[21]:


subs_df.head(2)


# In[22]:


# extract sub ids to link comments with parent submissions
sub_ids = comments_clean_df['link_id'].str.extract(r"t3_(.*)")
comments_clean_df['sub_id'] = sub_ids


# In[23]:


comments_clean_df.head(2)
# 1.0 is a direct response, NaN means that the comment is a reply to another comment


# In[24]:


subs_df = subs_df.drop(['Unnamed: 0', 'created_utc', 'over_18'], axis=1)


# In[25]:


# drop unnecessary columns
comments_clean_df = comments_clean_df.drop(['Unnamed: 0', 'controversiality', 'created_utc', 'id', 'date', 'link_id'], axis=1)


# In[26]:


merged_df = subs_df.merge(comments_clean_df, left_on="id", right_on="sub_id", suffixes=['_sub', '_com'])


# In[27]:


merged_df.head(3)


# In[28]:


merged_df.head()


# In[29]:


merged_df.shape # same number of rows as the number of rows in the comments data frame


# In[30]:


merged_df[~merged_df['nest_level'].isna()].head(2)


# In[31]:


# only keep direct responses (FOR NOW)
# ALSO: Ignore the url for now in the sub
merged_df = merged_df[~merged_df['nest_level'].isna()]


# In[32]:


np.mean(merged_df['selftext'].isna())


# In[33]:


# fill subs with missing selftext with empty strings
merged_df['selftext'] = merged_df['selftext'].fillna('')


# In[34]:


merged_df['subtext'] = merged_df['title'].str.cat(merged_df['selftext'], sep=' ')


# In[35]:


merged_df[merged_df['selftext'] != ''].head(2)


# In[36]:


subs = merged_df[['subtext']]
comments = merged_df[['body']]


# In[37]:


comments = comments_clean_df[['body']]
subs = subs_df[['title']]

# shuffle comments
size = comments.shape[0]
comments = comments.iloc[np.random.choice(np.arange(size), size=size, replace=False)]
testSize = ((int) (size*0.9))
comments_train = comments.iloc[0:testSize] # small for now for testing purposes
comments_test = comments.iloc[testSize:]


# In[38]:


#comments_train.shape


# In[39]:


# let's just keep comments small for now (training data)
#comments = comments_train[comments_train['body'].apply(len) < 100]
# TODO: delete this cell later
comments = comments_train
comments_test;


# In[40]:


comments_hf = Dataset.from_pandas(comments) # convert from pandas df to the hugging face version

#added
test_hf = Dataset.from_pandas(comments_test)


# In[41]:


subs_hf = Dataset.from_pandas(subs) # convert from pandas df to the hugging face version


# In[42]:


comments_hf


# In[43]:


subs_hf


# In[44]:


# create the tokenizer for our data to fine tune gpt2 on
tokenizer = AutoTokenizer.from_pretrained('gpt2')


# In[45]:


type(tokenizer)


# In[46]:


tokenizer('I am Luke') # how to tokenize a sentence -- dict returned
# input ids -- each word is mapped to a int
# attention_mask -- whether the token is inluded in the context or not (1 yes 0 no)


# In[47]:


tokenizer('I I I am am a Human')


# In[48]:


tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# In[49]:


# create a funciton that tells how to handle the data for preprocessing
def tokenize_function(examples):
    # Add padding and truncation to ensure consistent length
    output = tokenizer(examples["body"], padding='max_length', truncation=True, max_length=128)
    output["labels"] = output["input_ids"].copy()
    return output


# Tokenization does not happen on the GPU btw, but model training does!

# In[50]:


tokenized_comments = comments_hf.map(tokenize_function, batched=True) # batched = True


# In[51]:


len(tokenized_comments['input_ids'][2]) # every input has a LOT of paddidng :(


# In[52]:


tokenizer.model_max_length # oh we have PLENTY of space lol


# In[53]:


# tokenized comments is a dictionary of lists
# body, input_ids, attention
# body is the original text
len(tokenized_comments['input_ids'][0])


# In[54]:


len(tokenized_comments['body'][0].split(' '))


# In[55]:


np.array([len(ids) for ids in tokenized_comments['input_ids']]).max()


# In[56]:


tokenized_comments


# In[57]:


print(tokenized_comments.format['type'])


# In[58]:


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')


# In[57]:


# pad the data so that all the tensors in each batch are the same


# In[60]:


#amples = tokenized_comments[:2]
#samples = {k: v for k, v in samples.items()}
#[len(x) for x in samples["input_ids"]] # seems like the padding isn't great (too big!)

samples = tokenized_comments[:10]
samples = {k: v for k, v in samples.items() if k not in ['body']}#if k not in ["idx", "sentence1", "sentence2"]
[len(x) for x in samples["input_ids"]]


# In[61]:


batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}


# In[60]:


batch['input_ids'].shape


# In[61]:


batch['input_ids'][2] # here we can clearly see that we have right sided padding


# 

# In[62]:


# ! pip install --upgrade accelerate


# In[63]:


# ! pip uninstall -y transformers accelerate
# ! pip install transformers accelerate
# use to resolve an issue in hf 
# https://stackoverflow.com/questions/76225595/nameerror-name-partialstate-is-not-defined-error-while-training-hugging-face


# In[64]:


from transformers import TrainingArguments
get_ipython().system('pip install --upgrade transformers')
#need this here or get an error about partial states

training_args = TrainingArguments("test-trainer")


# In[65]:


get_ipython().system(' pip install transformers accelerate')


# In[66]:


model = GPT2LMHeadModel.from_pretrained('gpt2')


# In[67]:


#np.unique([len(i) for i in tokenized_comments['input_ids']], return_counts=True)


# In[68]:


tokenized_comments = tokenized_comments.remove_columns(["body"])


# In[69]:


model.resize_token_embeddings(len(tokenizer))


# In[70]:


trainer = Trainer(
    model=model,  # The pre-trained GPT-2 model
    args=training_args,  # Training arguments
    train_dataset=tokenized_comments,  # Tokenized and formatted training dataset
    data_collator=data_collator, 
    tokenizer=tokenizer
)


# In[ ]:


trainer.train()


# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# ADDRESSING THE INDEX ERROR
# * the lengths don't seem to be an issue! could it be a vocab issue?

# In[ ]:


# evaluate the perplexity of the model (a measure of predictiveness)
import math

eval_results = trainer.evaluate()

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# TODO: create evaluation and test dataset


# In[ ]:


867.9885 / 60


# In[ ]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[ ]:


# # save model
model_number = '1'
model.save_pretrained('/Model' + model_number + '/')


# In[ ]:


# save tokenizer 
tokenizer.save_pretrained('/Tokenizer' + model_number + '/')


# In[ ]:


# read model from file
model = GPT2LMHeadModel.from_pretrained('/Model' + model_number + '/')


# In[ ]:


#tokenizer = AutoTokenizer.from_pretrained('/content/gdrive/My Drive/chatbot/Tokenizer' + model_number + '/')


# In[ ]:





# Check if the model actually works

# In[ ]:


from transformers import pipelines


# In[ ]:


# Input text
input_text = "i'm"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2,
                        do_sample=True, temperature=0.7)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)


# In[ ]:


comments['body'].iloc[19]


# In[ ]:




