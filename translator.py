import os
from googletrans import Translator
from gtts import gTTS
import pygame
import time

def translate_and_speak(text, target_language):
    # Initialize the translator
    translator = Translator()

    # Translate the text
    translation = translator.translate(text, dest=target_language)
    translated_text = translation.text
    print(f"Translated Text: {translated_text}")

    # Convert translated text to speech
    tts = gTTS(translated_text, lang=target_language)
    
    # Save the audio file
    audio_file = 'translated_speech.mp3'
    tts.save(audio_file)

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and play the audio file
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Wait for the audio to finish
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  # Sleep for a short time to avoid busy waiting

    # Stop and unload the audio file
    pygame.mixer.music.unload()

    # Remove the audio file after playing
    os.remove(audio_file)


text_to_translate = "Hello" # get the text in here that will be generated via our model
target_language = 'mr'      # get the language from the 1st execution gui



# 'hi' is the language code for Hindi
# 'bn' is the language code for Bengali
# 'gu' is the language code for Gujrati
# 'kn' is the language code for Kannada
# 'ml' is the language code for Malayalam
# 'mr' is the language code for Marathi
# 'pa' is the language code for Punjabi
# 'ta' is the language code for Tamil
# 'te' is the language code for Telugu
# 'ur' is the language code for Urdu

translate_and_speak(text_to_translate, target_language)
