import pandas as pd
import numpy as np
import sklearn


import spacy
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
from pathlib import Path


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
import collections
import yaml
import pickle
import streamlit as st
import imgkit


with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)


train = pd.read_pickle("../objects/train.pkl")


processed = pd.read_pickle("../objects/processed.pkl")


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""



def extract_hardware(user_input, visualize=False):
    
    hardware_nlp = spacy.load("../hardware_train/model-last/")
    doc = hardware_nlp(user_input)

    extracted_entities = []

    
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

    
    if visualize == True:
        
        colors = {"HARDWARE": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["HARDWARE"], "colors": colors}
        
        html = displacy.render(doc, style="ent", options=options)

        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    return extracted_entities


def extract_app(user_input, visualize=False):
   
    app_nlp = spacy.load("../app_train/model-last/")
    doc = app_nlp(user_input)

    extracted_entities = []


    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))


    if visualize == True:

        colors = {"APP": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["APP"], "colors": colors}
        html = displacy.render(doc, style="ent", options=options)

        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    return extracted_entities


def extract_default(user_input):
    pass



