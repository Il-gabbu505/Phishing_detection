from sklearn.metrics import accuracy_score, recall_score, f1_score

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy:", acc)
    print("Recall:", rec)
    print("F1-score:", f1)