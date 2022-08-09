import random


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

import pandas as pd
import numpy as np
import re
import collections
import yaml

class Actions:
    memory = {"hardware": [], "app": []}

    def __init__(self, startup):
        
        self.startup = startup

    
    def utter_greet(self):
        
        return random.choice(
            [
                "Hi! My name is Q w Q. How may I assist you today?",
                "Hello. How may I be of help?",
            ]
        )

    
    def utter_goodbye(self):
        reaffirm = ["Is there anything else I could help you with?"]
        goodbye = [
            "Thank you for your time. Have a nice day!",
            "Glad I could be of help, have a nice day!",
        ]
        return random.choice(goodbye)

    
    def link_to_human(self):
        return random.choice(["Alright. Let me direct you to a representative!"])

    def battery(self, entity):
        if entity == "none":
            return random.choice(
                ["Can you tell me what device are you using?", "May I know what device you are using?"]
            )
        elif entity=="iphone":
            return("I understand your problem, please go to https://support.apple.com/en-in/iphone/repair/service/battery-power")
        
        elif entity=="macbook" or entity=="macbook pro" or entity=="mac":
            return("I see, please go to https://support.apple.com/mac/repair/service")

        
        elif entity=="airpods" or entity=="airpod":
            return("I'm sorry to hear that, please go to https://support.apple.com/en-in/airpods/repair/service")

        else:
            return random.choice(
                [
                    "I'm sorry to hear about that. You can check the battery health in your\
                                  settings. \nIf it is below 75%, please consider getting it replaced at your local apple store"
                ]
            )

    def forgot_pass(self):
        
        return("I'm sorry to hear about that, go to this [link](https://support.apple.com/en-in/HT201355)")

    def payment(self):
        return random.choice(
            ["Login with your Apple ID and update your payment method"]
        )

    def challenge_robot(self):
        return random.choice(
            [
                "I am Q w Q, your friendly neighbourhood personal assitant, and I am Batman.",
            ]
        )

    def update(self, entity):
        
        if entity=="iphone" or "iPhone":
            return random.choice(["Find details on how to update your iphone [here](https://support.apple.com/en-in/guide/iphone/iph3e504502/ios)"])
        if entity == "none":
            return random.choice(
                ["What device are you using?", "May I know what device you are using?"]
            )
        if entity == "macbook pro" or entity =="macbook":
            return random.choice(
                [
                    "Find details on how to update your macbook [here](https://support.apple.com/en-us/HT201541)"
                ]
            )
        if entity=="airpods" or entity=="airpod":
            return random.choice(["If you're setting up your AirPods for the first time, make sure that you have an iPhone with the latest version of iOS. And please refer the link : https://support.apple.com/en-in/HT207010"])
        else:
            return random.choice(
                [
                    "I'm sorry to hear that the update isn't working for you. Please find more information [here](https://support.apple.com/en-us/HT201222)"
                ]
            )

    def info(self, entity):
        if entity == "macbook pro" or entity==" macbook":
            return random.choice(
                [
                    "Okay! Right now we have 13 and 16 inch macbook pros. Please find more info [here](https://www.apple.com/macbook-pro/)"                ]
            )
        if entity == "ipad":
            return random.choice(["We have a number of options for iPads. You can check them out [here](https://www.apple.com/in/ipad/?afid=p238%7CssHgpnJQ7-dc_mtid_187079nc38483_pcrid_593187214049_pgrid_114045531295_pntwk_g_pchan__pexid__&cid=aos-IN-kwgo-ipad--slid---product-) "])
        if entity == "iphone":
            return random.choice(
                [
                    "Our most latest iPhone model is the iPhone 13 Pro. It comes in different model sizes. Please find more info [here]( https://www.apple.com/in/iphone/)"
                ]
            )
        if entity=="airpod" or entity=="airpods":
            return random.choice(
                [
                    "Airpods comes in various models. You can check them out here: https://www.apple.com/in/airpods/"
                ]
            )
        if entity == "none":
            return("What would you like to get info on good sir?")

    def fallback(self):
        return random.choice(
            [
                "I apologize. I didn't quite understand what you tried to say. Could you rephrase?"
            ]
        )

    def location(self,entity):
        if entity == "none":
            return random.choice(["I apologize. This is still work in progress."])
        else:
            return random.choice(
                [
                    "I apologize. This is still work in progress."
                ]
            )

    def repair(self,entity):
        if entity=="iphone":
            return("You can get Apple-certified repairs and service at one of our Apple Authorized Service Providers.\nHardware service may no longer be available for older iPhones. Contact your local provider to inquire about available service options.\nFor more information please check this [link](https://support.apple.com/en-in/iphone/repair/service)")
        if entity=="macbook" or entity=="macbook pro":
            return("To get service for your Mac, you can make a reservation at an Apple Authorized Service Provider.\n Make sure you know your Apple ID and password before your appointment.\n Depending on where you get service, you might be able to [check the status of your repair](https://idmsa.apple.com/IDMSWebAuth/signin?appIdKey=d7abc4ccb9b9f72d2f98c8d82fb9948668d09380a40c0fa64007a906a7502b4f&path=/) online.\n For more information please refer [here](https://support.apple.com/en-in/mac/repair/service)")
        if entity=="ipad":
            return("Apple offers many ways to get support and service for your iPad. Choose the one that’s best for you. \nOur support articles might answer your question. \nOr we can help you with issues like cracked screens or battery service.\n For more information please check this [link](https://support.apple.com/en-in/ipad/repair/service)")
        if entity=="airpods" or entity=="airpod":
            return("Set up an appointment at an Apple Authorized Service Provider. \nYou’ll need your serial number, so be sure to bring your Charging Case, proof of purchase, or original AirPods box along with your original AirPods.\n The serial number of your case is on the underside of the lid. \nIf your AirPods are connected to your iOS device, you can also go to Settings > General > About > AirPods for the serial number. \n For more information please check [here](https://support.apple.com/en-in/airpods/repair/service)")
        else:
            return("Im sorry but can you please state your query with the name of your device? ")
    def replace(self,entity):
        if entity=="iphone":
            return("You can get Apple-certified repairs and service at one of our Apple Authorized Service Providers.\nHardware service may no longer be available for older iPhones. Contact your local provider to inquire about available service options.\nFor more information please check this [link](https://support.apple.com/en-in/iphone/repair/service)")
        if entity=="macbook" or entity=="macbook pro":
            return("To get service for your Mac, you can make a reservation at an Apple Authorized Service Provider.\n Make sure you know your Apple ID and password before your appointment.\n Depending on where you get service, you might be able to [check the status of your repair](https://idmsa.apple.com/IDMSWebAuth/signin?appIdKey=d7abc4ccb9b9f72d2f98c8d82fb9948668d09380a40c0fa64007a906a7502b4f&path=/) online.\n For more information please refer [here](https://support.apple.com/en-in/mac/repair/service)")
        if entity=="ipad":
            return("Apple offers many ways to get support and service for your iPad. Choose the one that’s best for you. \nOur support articles might answer your question. \nOr we can help you with issues like cracked screens or battery service.\n For more information please check this [link](https://support.apple.com/en-in/ipad/repair/service)")
        if entity=="airpods" or entity=="airpod":
            return("Set up an appointment at an Apple Authorized Service Provider. \nYou’ll need your serial number, so be sure to bring your Charging Case, proof of purchase, or original AirPods box along with your original AirPods.\n The serial number of your case is on the underside of the lid. \nIf your AirPods are connected to your iOS device, you can also go to Settings > General > About > AirPods for the serial number. \n For more information please check [here](https://support.apple.com/en-in/airpods/repair/service)")
        else:
            return("Im sorry but can you please state your query with the name of your device?")