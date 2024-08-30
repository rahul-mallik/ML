import os
from googletrans import Translator
from gtts import gTTS
import pygame
import time
import tkinter as tk
from tkinter import ttk, messagebox

def translate_and_speak(text, target_language):
    # Initialize the translator
    translator = Translator()
    
    # Define supported languages
    supported_languages = {
        'Hindi': 'hi', 'Bengali': 'bn', 'Gujarati': 'gu',
        'Kannada' : 'kn', 'Malayalam':'ml' ,'Marathi' :'mr' ,
        'Punjabi':'pa' , 'Tamil':'ta' , 'Telugu':'te' , 'Urdu':'ur' 
    }

    # Check if the target language is supported
    if target_language not in supported_languages:
        raise ValueError("Invalid destination language")

    # Translate the text
    translation = translator.translate(text, dest=supported_languages[target_language])
    translated_text = translation.text
    print(f"Translated Text: {translated_text}")

    # Convert translated text to speech
    tts = gTTS(translated_text, lang=supported_languages[target_language])
    
    # Save the audio file
    audio_file = 'translated_speech.mp3'
    tts.save(audio_file)

    # Initialize pygame mixer
    pygame.mixer.init()

    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Wait for the audio to finish
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  # Sleep for a short time to avoid busy waiting

    # Stop and unload the audio file
    pygame.mixer.music.unload()

    # Remove the audio file after playing
    os.remove(audio_file)



def on_translate_button_click():
    text_to_translate = text_entry.get()
    target_language = language_combobox.get()

    if not text_to_translate:
        messagebox.showerror("Error", "Please enter some text to translate.")
        return
    if not target_language:
        messagebox.showerror("Error", "Please select a target language.")
        return

    try:
        translate_and_speak(text_to_translate, target_language)
    except Exception as e:
        messagebox.showerror("Error", str(e))


app = tk.Tk()
app.title("Text Translator and Speech Synthesizer")

# TExt box
tk.Label(app, text="Enter text to translate:").grid(row=0, column=0, padx=10, pady=10)
text_entry = tk.Entry(app, width=40)
text_entry.grid(row=0, column=1, padx=10, pady=10)

# Language Dropdown Menu
tk.Label(app, text="Select target language:").grid(row=1, column=0, padx=10, pady=10)
language_combobox = ttk.Combobox(app, values=[
    ('Hindi'), ('Bengali'), ('Gujarati'), 
    ('Kannada'), ('Malayalam'), ('Marathi'), 
    ('Punjabi'), ('Tamil'), ('Telugu'), ('Urdu')
])

language_combobox.grid(row=1, column=1, padx=10, pady=10)
language_combobox.set('Hindi')  # Default selection

# Translate button
translate_button = tk.Button(app, text="Translate and Speak", command=on_translate_button_click)
translate_button.grid(row=2, columnspan=2, pady=20)

# Run the application
app.mainloop()