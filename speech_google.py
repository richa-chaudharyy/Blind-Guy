# speech_google.py

import speech_recognition as sr
import pyttsx3

tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def speech_to_text_google():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        speak("Please start speaking.")
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("Captured Text:", text)
        speak("Captured your request.")
        return text
    except Exception as e:
        print("Error:", e)
        speak("I could not understand.")
        return None
