from flask import Flask, render_template, request
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords

# Download the stopwords resource if not already available
nltk.download('stopwords')

# Define the custom analyzer function as used in training
def process(text):
    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # Tokenize and remove stopwords
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

app = Flask(__name__)

# Custom Unpickler to ensure the process function is available
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'process':
            return process
        return super().find_class(module, name)

# Load the vectorizer and model (ensure paths are correct)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = CustomUnpickler(f).load()

with open("model.pkl", "rb") as f:
    model = CustomUnpickler(f).load()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    user_input = None
    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            # Transform the input text using the loaded vectorizer
            features = vectorizer.transform([user_input])
            pred = model.predict(features)
            prediction = "Spam" if pred[0] == 1 else "Ham"
    return render_template("index.html", prediction=prediction, user_input=user_input)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)