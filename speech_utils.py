import pyttsx3

tts_engine = pyttsx3.init()

def speak(text):
    print("Speaking:", text)  # for debugging
    tts_engine.say(text)
    tts_engine.runAndWait()
