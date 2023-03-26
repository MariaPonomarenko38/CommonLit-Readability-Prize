from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json

BASE_DIR = Path(__file__).resolve(strict=True).parent

if __name__ == '__main__':
    
    train_data = pd.read_csv(f"{BASE_DIR}/../input/train.csv")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_data['excerpt'])

    X_train, X_test, y_train, y_test = train_test_split(X, list(train_data['target']), test_size=0.15, shuffle=True, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    rms = mean_squared_error(y_train, reg.predict(X_train), squared=False)
    train_score = reg.score(X_train, y_train)
    test_score = reg.score(X_test, y_test)
    metrics = {'rmse': rms, 'train_score': train_score, 'test_score':test_score}
  
    with open('metrics_linear_reg.json', 'w') as f:
        json.dump(metrics, f)
    
    with open('../linear_regression_model1.pkl', 'wb') as file:
        pickle.dump(reg, file)

    with open('../count_vectorizer1.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)