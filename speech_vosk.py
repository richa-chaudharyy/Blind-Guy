from vosk import Model, KaldiRecognizer
import pyaudio
import json
import pyttsx3

# Initialize TTS
tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# VOSK speech-to-text function
def speech_to_text_vosk():
    model = Model("vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    speak("Please start speaking.")
    print("Listening...")

    while True:
        data = stream.read(4096, exception_on_overflow=False)

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            print("Internal text captured:", text)
            speak("Captured your request.")
            return text

# Test run
if __name__ == "__main__":
    result = speech_to_text_vosk()
    print("Output Text:", result)
