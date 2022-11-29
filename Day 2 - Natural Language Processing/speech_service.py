import speech_recognition as sr
import pyttsx3

# Speech to text function - you need to provide your microphone sample rate, chunk_size and device index of your microphone
def speech_to_text(sample_rate, chunk_size, index):
    # Set American English
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=sample_rate, chunk_size=chunk_size, device_index=index) as source:
        # Adjusts the energy threshold dynamically using audio from source (an AudioSource instance) to account for ambient noise.
        print("Please wait one second for calibrating microphone...")
        r.pause_threshold = 0.8
        r.dynamic_energy_threshold = True
        r.adjust_for_ambient_noise(source, duration=1)
        print("Ok, microphone is ready...")
        audio = r.listen(source, timeout=None)
        transcript = ""
        try:
            transcript = r.recognize_google(audio, language="en-US")
            print('You: ' + transcript)
        except:
            print('VA: I did not hear anything....')

    return transcript.lower()


# Text to Speech function - you need to provide the text as parameter
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()