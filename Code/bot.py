import streamlit as st
from com import listen
from actions import Actions
from ner import extract_app, extract_hardware
from initialize_intent_classifier import infer_intent
import pandas as pd
import numpy as np
import yaml
import pyttsx3
from streamlit import caching


import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process

st.set_option('deprecation.showPyplotGlobalUse', False)

def speak(text):
    engine = pyttsx3.init()
    try:
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except RuntimeError:
        engine.stop()


global temp_count
temp_count=0
with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)


train = pd.read_pickle("../objects/train.pkl")

sns.set(style="ticks", color_codes=True)
sns.set_style(style="whitegrid")


respond = lambda response: f" Q w Q: {response}"


def main(phrase="Tell  Q w Q  something!"):

    
    a = Actions(phrase)

  


    intents, user_input, history_df, end = conversation(a)
    


    if end == False:
        st.experimental_singleton.clear()
        
        conversation(Actions("Could you please rephrase?"))


def conversation(starter):
    """ Represents one entire flow of a conversation that takes in the Actions 
    object to know what prompt to start with """

    a = starter

    user_input, hardware, app, intents, history_df = talk(prompt=a.startup)


    max_intent, extracted_entities = action_mapper(history_df)
    print(max_intent)
    if extracted_entities != []:
        if len(extracted_entities) == 1:
            entity = extracted_entities[0]
            print(f"Found 1 entity: {entity}")
        elif len(extracted_entities) == 2:
            entity = extracted_entities[:2]
            print(f"Found 2 entities: {entity}")
    else:
        entity = None

    end = listener(max_intent, entity, a)

    return (intents, user_input, history_df, end)


def talk(prompt):
    """ Goes through an initiates a conversation and returns:
    
    User_input: string
    Hardware: List of strings containing entities extracted
    App: List of strings containing entities extracted
    Intents: Tuple that can be unpacked to:
        - User_input
        - Predictions: Dictionary containing intents as keys and prediction probabilities (0-1) as values
    History_df: Dialogue state given the input

     """
    user_input = st.text_input(prompt)
    if st.button("Mic"):
        user_input = listen()
        st.text("You said : " +user_input)

  
    intents, hardware, app = initialize(user_input)
    user_input, prediction = intents

   
    columns = entities["hardware"] + entities["apps"] + list(prediction.keys())
    history_df = pd.DataFrame(dict(zip(columns, np.zeros(len(columns)))), index=[0])

    history_df = history_df.append(to_row(prediction, hardware, app), ignore_index=True)

    return (user_input, hardware, app, intents, history_df)


def listener(max_intent, entity, actions):
    """ Takes in dialogue state and maps that to a response"""


    def follow_up(prompt="Could you please rephrase?"):
        """ Business logic for following up """


        end = None

        st.text("Did that solve your problem?")
        yes = st.button("Yes")
        no = st.button("No")

        if yes:
            st.text(respond("Great! Glad I was able to be of service to you!"))
            end = True

        if no:
            global temp_count
            temp_count+=1
            if temp_count==3:
                st.text(respond(a.link_to_human()))
                temp=a.link_to_human()
                speak(temp)
                st.image("images/representative.png")

            end = True
        return end


    a = actions


    end = None

    if max_intent == "greeting":
        st.write(respond(a.utter_greet()))
        temp=a.utter_greet()
        speak(temp)
        end = follow_up()
    elif max_intent=="repair":
        st.write(respond(a.repair(entity)))
        temp=a.repair(entity)
        speak(temp)
        end = follow_up()
    elif max_intent=="lost_replace":
        st.write(respond(a.replace(entity)))
        temp=a.replace(entity)
        speak(temp)
        end = follow_up()
    elif max_intent == "info":
        st.write(respond(a.info(entity)))
        temp=a.info(entity)
        speak(temp)
        end = follow_up()
    elif max_intent == "update":
        st.write(respond(a.update(entity)))
        temp=a.update(entity)
        speak(temp)
        end = follow_up()
    elif max_intent == "forgot_password":
        st.write(respond(a.forgot_pass()))
        temp=a.forgot_pass()
        speak(temp)
 
        end = follow_up()
    elif max_intent == "challenge_robot":
        st.write(respond(a.challenge_robot()))
        temp=a.challenge_robot()
        speak(temp)

    elif max_intent == "goodbye":
        st.write(respond(a.utter_goodbye()))
        temp=a.utter_goodbye()
        speak(temp)
        st.image("images/eve-bye.jpg", width=400)
        st.text("Q w Q waves you goodbye!")
    elif max_intent == "payment":
        st.write(respond(a.payment()))
        temp=a.payment()
        speak(temp)
        end = follow_up()
    elif max_intent == "speak_representative":
        st.write(respond(a.link_to_human()))
        temp=a.link_to_human()
        speak(temp)
        st.image("images/representative.png")
    elif max_intent == "battery":
        st.write(respond(a.battery(entity)))
        temp=a.battery(entity)
        speak(temp)
        end = follow_up()
    elif max_intent == "fallback":
        st.write(respond(a.fallback()))
        temp=a.fallback()
        speak(temp)
    elif max_intent == "location":
        st.write(respond(a.location(entity)))
        temp=a.location(entity)
        speak(temp)

    return end


def backend_dash(intents, user_input, history_df):
    """ Visualizes with a dashboard the entire dialogue state of a conversation given state params """

    st.subheader("Q w Q's Predictions")

    user_input, pred = intents
    pred = {k: round(float(v), 3) for k, v in pred.items()}


    g = sns.barplot(
        list(pred.keys()),
        list(pred.values()),
        palette=sns.cubehelix_palette(8, reverse=True),
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    st.pyplot(bbox_inches="tight")

    st.subheader("Hardware Entities Detected")
    hardware = extract_hardware(user_input, visualize=True)
    st.subheader("App Entities Detected")
    app = extract_app(user_input, visualize=True)

    st.subheader("Dialogue State History")
    st.dataframe(history_df)


def to_row(prediction, hardware, app):
    row = []


    if hardware == []:
        for i in range(len(entities["hardware"])):
            row.append(0)
    else:
        for entity in entities["hardware"]:
            if hardware[0][0] == entity:
                row.append(1)
            else:
                row.append(0)


    if app == []:
        for i in range(len(entities["apps"])):
            row.append(0)
    else:
        for entity in entities["apps"]:
            if app[0][0] == entity:
                row.append(1)
            else:
                row.append(0)

    
    for i in prediction.items():
        row.append(i[1])

    
    columns = entities["hardware"] + entities["apps"] + list(prediction.keys())
    df = pd.DataFrame(dict(zip(columns, row)), index=[0])

    return df


def action_mapper(history_df):
    """ Simply maps an history state to:
    
    A max intent: String
    Entities: List of entities extracted
    
    """
    prediction_probs = history_df.iloc[-1:, -len(set(train["Intent"])) :]
    predictions = [
        *zip(list(prediction_probs.columns), list(prediction_probs.values[0]))
    ]

    
    entities = history_df.iloc[-1:, : -len(set(train["Intent"]))]
    mask = [True if i == 1.0 else False for i in list(entities.values[0])]
    extracted_entities = [b for a, b in zip(mask, list(entities.columns)) if a]

    
    predictions.sort(key=lambda x: x[1])
    
    max_tuple = predictions[-1:]
    
    
    max_intent = max_tuple[0][0]
    print(f'max_intent{max_intent}')

    

    return (max_intent, extracted_entities)


def initialize(user_input):
    """ Takes the user input and returns the entity representation and predicted intent"""
    
    intents = infer_intent(user_input)

    
    hardware = extract_hardware(user_input)
    app = extract_app(user_input)

    if hardware == []:
        hardware = "none"

    if app == []:
        app = "none"

    return (intents, hardware, app)


if __name__ == "__main__":
    main()
