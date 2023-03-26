from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve(strict=True).parent

    train_data = pd.read_csv(f"{BASE_DIR}/../input/train.csv")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_data['excerpt'])
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    prepared_docs_for_tokenizer = train_data['excerpt'].apply(lambda x: "[CLS] " + x + " [SEP]")
    bert_tokenized_docs = prepared_docs_for_tokenizer.apply(lambda x: tokenizer.tokenize(x))
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(bert_tokenized_docs, show_progress_bar=True)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, list(train_data['target']), test_size=0.15, shuffle=True, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    rms = mean_squared_error(y_train, reg.predict(X_train), squared=False)
    with open('linear_regression_model.pkl', 'wb') as file:
        pickle.dump(reg, file)

    with open('count_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    print(rms)