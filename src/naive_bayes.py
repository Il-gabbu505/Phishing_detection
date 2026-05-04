from sklearn.naive_bayes import MultinomialNB

def train_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model