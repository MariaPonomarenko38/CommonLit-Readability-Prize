
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import pickle

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/linear_regression_model.pkl", 'rb') as file:
    model = pickle.load(file)

with open(f"{BASE_DIR}/count_vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

def predict(text):
    X_new = vectorizer.transform([text])
    y_pred = model.predict(X_new)
    return y_pred[0]

if __name__ == '__main__':
    print(predict.__file__)