
import streamlit as st
import sklearn 
import pandas as pd
import numpy as np


import bot


def main():

    st.sidebar.image("images/logo-wo-slogan.jpeg", width=200)


    selected = st.sidebar.radio(
        "Navigate pages", options=["Home", "Keyword Exploration"]
    )

    if selected == "Home":
        home()




def home():
    st.title("Welcome, we are happy to serve you.")
    bot.main()


if __name__ == "__main__":
    main()
