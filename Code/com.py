import speech_recognition as sr

def listen():
    r = sr.Recognizer()

    
    with sr.Microphone() as source:
       
        
        while True:
            audio = r.listen(source)

            
            google = r.recognize_google(audio)
            return google
