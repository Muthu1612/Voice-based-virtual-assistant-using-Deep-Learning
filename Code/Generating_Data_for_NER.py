# %%

import pandas as pd
print(f"Pandas: {pd.__version__}")
import numpy as np
print(f"Numpy: {np.__version__}")

import tensorflow as tf
print(f"Tensorflow: {tf.__version__}")
from tensorflow import keras
print(f"Keras: {keras.__version__}")
import sklearn
print(f"Sklearn: {sklearn.__version__}")


import spacy
print(f'spaCy: {spacy.__version__}')
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
import plac
from pathlib import Path


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)


from tqdm.notebook import tqdm
tqdm().pandas() 

import collections
import yaml
import pickle


with open(r'objects/intents.yml') as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)


from tqdm.notebook import tqdm
tqdm().pandas() 

from IPython.core.display import display, HTML
    

train = pd.read_pickle('objects/train.pkl')

print(train.head())
print(f'\nintents:\n{intents}')

processed = pd.read_pickle('objects/processed.pkl')

# %%


entities = {'hardware': ['macbook pro', 'iphone', 'iphones', 'mac',
        'ipad', 'watch', 'TV', 'airpods','macbook'],
    'apps':['app store', 'garageband', 'books', 'calendar',
           'podcasts', 'notes', 'icloud', 'music', 'messages',
           'facetime','catalina','maverick']}


with open('objects/entities.yml', 'w') as outfile:
    yaml.dump(entities, outfile, default_flow_style=False)

# %%
entities

# %%
def offsetter(lbl, doc, matchitem):
    ''' Converts word position to string position, because output of PhraseMatcher returns '''
    one = len(str(doc[0:matchitem[1]]))
    subdoc = doc[matchitem[1]:matchitem[2]]
    two = one + len(str(subdoc))
    

    if one != 0:
        one += 1
        two += 1
    return (one, two, lbl)


offsetter('HARDWARE', nlp('hmm macbooks are great'),(2271554079456360229, 1, 2))

# %%

nlp = spacy.load('en_core_web_sm')


if 'ner' not in nlp.pipe_names:

    nlp.add_pipe("ner")


def spacify_row(document, label, entity_keywords):

    matcher = PhraseMatcher(nlp.vocab)


    for i in entity_keywords:
        matcher.add(label, None, nlp(i))


    nlp_document = nlp(document)
    matches = matcher(nlp_document)
    

    entity_list = [offsetter(label, nlp_document, match) for match in matches]
    

    return (document, {'entities': entity_list})

# %%

string_utterance = processed['Processed Inbound'].progress_apply(" ".join)


spacify_row('I love my macbook and my iphone', 'HARDWARE', 
            entity_keywords = entities.get('hardware'))

# %%
entity_train = string_utterance.progress_apply(spacify_row,
                label = 'HARDWARE',              
                entity_keywords = entities.get('hardware'))

# %%

hardware_train = [(i,j) for i,j in entity_train if j['entities'] != []]


print(f'{len(hardware_train)} out of {len(entity_train)} Tweets contain a hardware entity')


pickle_out = open('objects/hardware_train.pkl', 'wb')
pickle.dump(hardware_train, pickle_out)

# %%
entity_train = string_utterance.progress_apply(spacify_row,
                label = 'APP',              
                entity_keywords = entities.get('apps'))


app_train = [(i,j) for i,j in entity_train if j['entities'] != []]


pickle_out = open('objects/app_train.pkl', 'wb')
pickle.dump(app_train, pickle_out)


print(f'{len(app_train)} out of {len(entity_train)} Tweets contain an app entity')

# %%
hardware_train[:5]

# %%
app_train[:5]

# %%
for _, annotations in hardware_train:
        for ent in annotations.get('entities'):
                print(ent)


