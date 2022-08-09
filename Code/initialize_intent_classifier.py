import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from keras.models import load_model

import pandas as pd
import numpy as np
import yaml

train = pd.read_pickle("../objects/train.pkl")
print(f"Training data: {train.head()}")

model = load_model("../models/intent_classification.h5")


X_train, X_val, y_train, y_val = train_test_split(
    train["Utterance"],
    train["Intent"],
    test_size=0.3,
    shuffle=True,
    stratify=train["Intent"],
    random_state=7,
)



le = LabelEncoder()
le.fit(y_train)


t = Tokenizer()
t.fit_on_texts(X_train)


max_length = len(max(X_train, key=len))



def convert_to_padded(tokenizer, docs):
    """ Taking in Keras API Tokenizer and documents and returns their padded version """

    embedded = tokenizer.texts_to_sequences(docs)

    padded = pad_sequences(embedded, maxlen=max_length, padding="post")
    return padded


padded_X_train = convert_to_padded(tokenizer=t, docs=X_train)
padded_X_val = convert_to_padded(tokenizer=t, docs=X_val)


def infer_intent(user_input):
    """ Making a function that recieves a user input and outputs a 
    dictionary of predictions """
    assert isinstance(user_input, str), "User input must be a string!"
    keras_input = [user_input]
    print(user_input)

    padded_text = convert_to_padded(t, keras_input)

    x = padded_text

    probs = model.predict(padded_text)

    classes = le.classes_


    predictions = dict(zip(classes, probs[0]))
    sorted_predictions = {
        k: v
        for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    }


    with open("../objects/sorted_predictions.yml", "w") as outfile:
        yaml.dump(sorted_predictions, outfile, default_flow_style=False)

    return user_input, sorted_predictions



