from src.preprocessing import load_data, split_data, tfidf_features, lstm_tokenizer
from src.naive_bayes import train_nb
from src.lstm_model import build_lstm
from src.evaluate import evaluate
import numpy as np


df = load_data("data/phishing_dataset.csv")


X_train, X_test, y_train, y_test = split_data(df)


X_train_nb, X_test_nb = tfidf_features(X_train, X_test)
nb_model = train_nb(X_train_nb, y_train)

y_pred_nb = nb_model.predict(X_test_nb)
print("\nNaïve Bayes Results:")
evaluate(y_test, y_pred_nb, "Naive Bayes")



X_train_lstm, X_test_lstm = lstm_tokenizer(X_train, X_test)

lstm_model = build_lstm()
lstm_model.fit(X_train_lstm, y_train, epochs=3, batch_size=32)

y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32")

print("\nLSTM Results:")
evaluate(y_test, y_pred_lstm, "LSTM")