import re, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())      # lowercase + special char removal
    tokens = word_tokenize(text)                      # tokenization
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    lemma = [lemmatizer.lemmatize(w) for w in tokens] # lemmatization
    stem = [stemmer.stem(w) for w in tokens]          # stemming

    return lemma, stem

# Example text
text = "Natural Language Processing (NLP) is exciting!!! 😊 2025"

lemma_out, stem_out = preprocess(text)

print("Lemmatized:", lemma_out)
print("Stemmed:", stem_out)
