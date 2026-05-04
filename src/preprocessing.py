import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def load_data(path):
    df = pd.read_csv(path)

    df['text'] = df['subject'].astype(str) + " " + \
                 df['sender'].astype(str) + " " + \
                 df['email_text'].astype(str)

    df['label'] = df['label'].map({
        'legitimate': 0,
        'phishing': 1
    })

    df['text'] = df['text'].apply(clean_text)

    return df[['text', 'label']]

def split_data(df):
    return train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

def tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def lstm_tokenizer(X_train, X_test):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, maxlen=100)
    X_test = pad_sequences(X_test, maxlen=100)

    return X_train, X_test