# %%
import spacy 

nlp = spacy.load('en_core_web_sm')
nlp.pipe_names

# %%
sentence = 'Daniil Medvedev and Novak Djokovic have built an intriguing rivalry since the Australian Open decider, which the Serb won comprehensively.'
doc = nlp(sentence)

from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)



# %%
[(X, X.ent_iob_, X.ent_type_) for X in doc if X.ent_type_]

# %%
spacy.explain("NORP")

# %%
import pandas as pd
train = pd.read_pickle('objects/train.pkl')

# %%
hardware_train =  pd.read_pickle('objects/hardware_train.pkl')
len(hardware_train)

# %%
app_train =  pd.read_pickle('objects/app_train.pkl')
app_train

# %%


# %%
# trainData= app_train
trainData= hardware_train

# %%
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(hardware_train, test_size = 0.2, random_state = 1)
# from sklearn.model_selection import train_test_split
# training_set, test_set = train_test_split(app_train, test_size = 0.2, random_state = 1)

# %%
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
nlp = spacy.blank("en")
db = DocBin() 
for text, annot in tqdm(test_set): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]: 
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    try:
        doc.ents = ents 
        db.add(doc)
    except:
        print(text, annot)
db.to_disk("hardware_train/dev.spacy") 

# %%
import spacy

nlp = spacy.load("app_train/model-last/")

# %%
sentence = """show music store phone like want store music phone."""

doc = nlp(sentence)

from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)

# %%
import spacy

nlp2 = spacy.load("hardware_train/model-last/")

# %%
sentence = """show music store iphone like want store music iphone."""

doc = nlp2(sentence)

from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)

# %%
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
print(f'gensim: {gensim.__version__}')


from nltk.tokenize import word_tokenize 
from nltk.tokenize import TweetTokenizer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

from tempfile import mkdtemp
import pickle
import joblib


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)


import os
import yaml
import collections
import scattertext as st
import math


from tqdm.notebook import tqdm
tqdm.pandas()


processed_inbound = pd.read_pickle('objects/processed_inbound.pkl')
processed = pd.read_pickle('objects/processed.pkl')


with open(r'objects/intents.yml') as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)


print(f'\nintents:\n{intents}')
print(f'\nprocessed:\n{processed.head()}')

# %%
processed_inbound.head(12)

# %%
import spacy 

nlp = spacy.load('en_core_web_sm')
nlp.pipe_names

# %%
sentence = 'Daniil Medvedev and Novak Djokovic have built an intriguing rivalry since the Australian Open decider, which the Serb won comprehensively.'
doc = nlp(sentence)

from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)

# %%
extracted_entities = []


for ent in doc.ents:
    extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

# %%
temp=[]
for sentence in processed_inbound:
    
    new_sentence=" ".join(sentence)
    temp.append(new_sentence)

# %%
temp

# %%
from spacy import displacy
count=0
extracted_entities = []
temp2=[]
for sentence in temp:
    
    doc = nlp2(sentence)
    


    
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        temp2.append(ent.text)


# %%
temp2

# %%


def top10_bagofwords(data, output_name, title):
    ''' Taking as input the data and plots the top 10 words based on counts in this text data'''
    bagofwords = CountVectorizer()
    inbound = bagofwords.fit_transform(data)
    inbound 
    word_counts = np.array(np.sum(inbound, axis=0)).reshape((-1,))
    words = np.array(bagofwords.get_feature_names_out())
    words_df = pd.DataFrame({"word":words, 
                             "count":word_counts})
    words_rank = words_df.sort_values(by="count", ascending=False)
    
    words_rank.head()
    
    plt.figure(figsize=(12,6))
    
   
    sns.barplot(x=words_rank['word'][:10], y=words_rank['count'][:10], palette = 'inferno')
    plt.title(title)
    
    
    plt.savefig(f'visualizations/{output_name}.png')
    
    plt.show()

# %%
top10_bagofwords(temp2, 'most_common_before' ,'Top 10 Most Common Words in My Inbound Data Before Preprocessing')


