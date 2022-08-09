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


processed_inbound = pd.read_pickle('objects/processed_inbound_extra.pkl')
processed = pd.read_pickle('objects/processed.pkl')


with open(r'objects/intents.yml') as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)


print(f'\nintents:\n{intents}')
print(f'\nprocessed:\n{processed.head()}')

# %%

ideal = {'Greeting': 'hi hello yo hey whats up howdy morning',
        'Update': 'have problem with update'}

ideal = {'battery': 'battery power charge', 
         'forgot_password': 'password account login',
         'payment': 'credit card payment pay transaction',
         'update': 'update upgrade',
         'info': 'info information know',
         'repair': 'repar fix broken',
         'lost_replace': 'replace lost gone missing trade'
         ,'location': 'nearest  location store'
        }

def add_extra(current_tokenized_data, extra_tweets):
    ''' Adding extra tweets to current tokenized data'''
    
    
    extra_tweets = pd.Series(extra_tweets)

    
    print('Converting to string...')
    string_processed_data = current_tokenized_data.progress_apply(" ".join)

    
    string_processed_data = pd.concat([string_processed_data, extra_tweets], axis = 0)

    
    tknzr = TweetTokenizer(strip_handles = True, reduce_len = True)

    return string_processed_data



# %%
processed_inbound_extra = add_extra(processed['Processed Inbound'], list(ideal.values()))


processed_inbound_extra.to_pickle('objects/processed_inbound_extra.pkl')

processed_inbound_extra

# %%
processed_inbound_extra[-7:]

# %%
processed_inbound_extra.shape

# %%
ideal

# %%
processed.shape

# %%
def train_doc2vec(string_data, max_epochs, vec_size, alpha):
    
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) 
                   for i, _d in enumerate(string_data)]
    
    
    model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples = model.corpus_count, epochs= model.epochs)
        
        model.alpha -= 0.0002
        
        model.min_alpha = model.alpha

    
    model.save("models/d2v.model")
    print("Model Saved")
    

train_doc2vec(processed_inbound_extra, max_epochs = 100, vec_size = 20, alpha = 0.025)

# %%

model = Doc2Vec.load("models/d2v.model")


inbound_d2v = np.array([model.dv[i] for i in range(processed_inbound_extra.shape[0])])


with open('objects/inbound_d2v.pkl', 'wb') as f:
    pickle.dump(inbound_d2v, f)

inbound_d2v

# %%
inbound_d2v.shape

# %%

intents_ideal = {'app': ['app', 'prob']}
inferred_vectors = []

for keywords in intents_ideal.values():
    inferred_vectors.append(model.infer_vector(keywords))
    
inferred_vectors

# %%
'hi hello yo hey whats up'.split()

# %%
ideal

# %%

intents_repr = {'Battery': ['io', 'drain', 'battery', 'iphone', 'twice', 'fast', 'io', 'help'],
    'Update': ['new', 'update', 'i️', 'make', 'sure', 'download', 'yesterday'],
    'iphone': ['instal', 'io', 'make', 'iphone', 'slow', 'work', 'properly', 'help'],
    'app': ['app', 'still', 'longer', 'able', 'control', 'lockscreen'],
    'mac': ['help','mac','app','store','open','can','not','update','macbook','pro','currently','run','o','x',
  'yosemite'], 'greeting': ['hi', 'hello', 'yo', 'hey', 'whats', 'up']
    }


# %%

tknzr = TweetTokenizer(strip_handles = True, reduce_len = True)

intents_repr = {k:tknzr.tokenize(v) for k, v in ideal.items()}
print(intents_repr)


with open('objects/intents_repr.yml', 'w') as outfile:
    yaml.dump(intents_repr, outfile, default_flow_style=False)


tags = []

tokenized_processed_inbound = processed_inbound.apply(tknzr.tokenize)

def report_index_loc(tweet, intent_name):
    ''' Takes in the Tweet to find the index for and returns a report of that Tweet index along with what the 
    representative Tweet looks like'''
    try:
        tweets = []
        for i,j in enumerate(tokenized_processed_inbound):
            if j == tweet:
                tweets.append((i, True))
            else:
                tweets.append((i, False))
        index = []
        get_index = [index.append(i[0]) if i[1] == True else False for i in tweets] # Comprehension saves space

        preview = processed_inbound.iloc[index]

        
        tags.append(str(index[0]))
    except IndexError as e:
        print('Index not in list, move on')
        return
        
    return intent_name, str(index[0]), preview


print('TAGGED INDEXES TO LOOK FOR')
for j,i in intents_repr.items():
    try:
        print('\n{} \nIndex: {}\nPreview: {}'.format(*report_index_loc(i,j)))
    except Exception as e:
        print('Index ended')


intents_tags = dict(zip(intents_repr.keys(), tags))
intents_tags

# %%
similar_doc = model.docvecs.most_similar('76062',topn = 1000)

similar_doc[:5]

# %%
similar_doc = model.docvecs.most_similar('76065',topn = 1000)
similar_doc

# %%
import nltk
from nltk.corpus import stopwords
stopwords.words('english').index('to')

# %%
intents_tags

# %%
model.docvecs.most_similar('10')

# %%
intents_tags

# %% [markdown]
# prompt the user for update or broken.

# %%
vals = [word_tokenize(tweet) for tweet in list(processed_inbound.iloc[[10,1]].values)]
vals

# %%
train = pd.DataFrame()
intent_indexes = {}


def generate_intent(target, itag):
    similar_doc = model.dv.most_similar(itag,topn = target)
    
    indexes = [int(i[0]) for i in similar_doc]
    intent_indexes[intent_name] = indexes

    return [word_tokenize(tweet) for tweet in list(processed_inbound.iloc[indexes].values)]


for intent_name, itag in intents_tags.items():
    train[intent_name] = generate_intent(1000, itag)


manually_added_intents = {
    'speak_representative': [['talk','human','please','person','someone real'],
                             ['let','me','talk','to','apple','support'], 
                             ['can','i','speak','agent','person']], 
    'greeting': [['hi'],['hello'], ['whats','up'], ['good','morning'],
                 ['good','evening'], ['good','night'],['yo'],['hii']],
    'goodbye': [['goodbye'],['bye'],['thank'],['thanks'], ['done'],['byeee']], 
    'challenge_robot': [['robot','human'], ['are','you','robot'],
                       ['who','are','you'],['I','do not','like','you']]
}



def insert_manually(target, prototype):
    ''' Taking a prototype tokenized document to repeat until
    you get length target'''
    factor = math.ceil(target / len(prototype))
    content = prototype * factor
    return [content[i] for i in range(target)]


for intent_name in manually_added_intents.keys():
    train[intent_name] = insert_manually(1000, [*manually_added_intents[intent_name]])



hybrid_intents = {'update':(300,700,[['want','update'], ['update','not','working'], 
                                     ['phone','need','update']], 
                            intents_tags['update']),
                  'info': (800,200, [['need','information'], 
                                       ['want','to','know','about'], ['what','are','macbook','stats'],
                                    ['any','info','next','release','?'],['tell','me','about','the','new']], 
                             intents_tags['info']),
                  'payment': (300,700, [['payment','not','through'], 
                                       ['iphone', 'apple', 'pay', 'but', 'not', 'arrive'],
                                       ['how','pay','for', 'this'],
                                       ['can','i','pay','for','this','first']], 
                             intents_tags['payment']),
                  'forgot_password': (600,400, [['forgot','my','pass'], ['forgot','my','login'
                                ,'details'], ['cannot','log','in','password'],['lost','account','recover','password']], 
                             intents_tags['forgot_password'])
                 }

def insert_hybrid(manual_target, generated_target, prototype, itag):
    return insert_manually(manual_target, prototype) + list(generate_intent(generated_target, itag))


for intent_name, args in hybrid_intents.items():
    train[intent_name] = insert_hybrid(*args)


neat_train = pd.DataFrame(train.T.unstack()).reset_index().iloc[:,1:].rename(columns={'level_1':'Intent', 0: 'Utterance'})

neat_train = neat_train[['Utterance','Intent']]


neat_train.to_pickle('objects/train.pkl')

show = lambda x: x.head(10).style.set_properties(**{'background-color': 'black',                                                   
                                    'color': 'lawngreen',                       
                                    'border-color': 'white'})\
.applymap(lambda x: f"color: {'lawngreen' if isinstance(x,str) else 'red'}")\
.background_gradient(cmap='Blues')

print(train.shape)
show(train)

# %%
print(neat_train.shape)
show(neat_train)

# %%
neat_train.tail(44)

# %%
processed.head(5)


