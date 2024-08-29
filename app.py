from flask import Flask, render_template, request
from googletrans import Translator, LANGUAGES

app = Flask(__name__)
translator = Translator()

# Home route to render HTML page
@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES)

# Route for handling the translation
@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['input_text']
    target_language = request.form['language']
    if input_text and target_language:
        translated_text = translator.translate(input_text, dest=target_language).text
    else:
        translated_text = "Please provide valid input."
    return render_template('index.html', translated_text=translated_text, languages=LANGUAGES)

if __name__ == '__main__':
    app.run(debug=True)
