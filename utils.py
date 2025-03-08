# filepath: c:\Users\deeja\Downloads\Compressed\Email-Spam-Detection-Using-NLP-main\dataset\utils.py
import string
from nltk.corpus import stopwords

def process(text):
    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # Tokenize and remove stopwords
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean