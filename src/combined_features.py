"""
综合方法: Word2Vec + TF-IDF特征融合 + XGBoost/Logistic Regression
目标: Kaggle AUC >= 0.94
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import os
import joblib

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")

# 最小停用词列表
MINIMAL_STOPWORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our',
    'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
])

def preprocess_text(raw_review, remove_stopwords=True):
    """Clean and tokenize text"""
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in MINIMAL_STOPWORDS]
    return words

def preprocess_text_string(raw_review, remove_stopwords=True):
    """Clean text and return as string for TF-IDF"""
    words = preprocess_text(raw_review, remove_stopwords)
    return ' '.join(words)

def get_mean_embedding(words, model, vector_size=300):
    """Calculate mean embedding"""
    feature_vec = np.zeros((vector_size,), dtype="float32")
    n_words = 0
    for word in words:
        if word in model.wv:
            feature_vec += model.wv[word]
            n_words += 1
    if n_words > 0:
        feature_vec /= n_words
    return feature_vec

def main():
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv/labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv/testData.tsv"),
                       header=0, delimiter="\t", quoting=3)

    # Preprocess
    print("Preprocessing reviews...")
    train_reviews = []
    train_texts = []
    for i, review in enumerate(train["review"]):
        if (i+1) % 5000 == 0:
            print(f"Processing train review {i+1}/{len(train)}")
        words = preprocess_text(review, remove_stopwords=True)
        train_reviews.append(words)
        train_texts.append(' '.join(words))

    test_reviews = []
    test_texts = []
    for i, review in enumerate(test["review"]):
        if (i+1) % 5000 == 0:
            print(f"Processing test review {i+1}/{len(test)}")
        words = preprocess_text(review, remove_stopwords=True)
        test_reviews.append(words)
        test_texts.append(' '.join(words))

    all_reviews = train_reviews + test_reviews

    # Train Word2Vec
    print("\nTraining Word2Vec...")
    w2v_model = Word2Vec(
        sentences=all_reviews,
        vector_size=300,
        window=10,
        min_count=2,
        workers=4,
        epochs=40,
        sg=1,
        sample=1e-3,
        negative=10
    )

    # Create Word2Vec embeddings
    print("Creating Word2Vec embeddings...")
    train_w2v = np.zeros((len(train_reviews), 300), dtype="float32")
    for i, words in enumerate(train_reviews):
        train_w2v[i] = get_mean_embedding(words, w2v_model, 300)

    test_w2v = np.zeros((len(test_reviews), 300), dtype="float32")
    for i, words in enumerate(test_reviews):
        test_w2v[i] = get_mean_embedding(words, w2v_model, 300)

    # Create TF-IDF features
    print("\nCreating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)

    print(f"TF-IDF shape: train={train_tfidf.shape}, test={test_tfidf.shape}")

    # Combine features
    print("\nCombining features...")
    # Scale Word2Vec features
    scaler = StandardScaler()
    train_w2v_scaled = scaler.fit_transform(train_w2v)
    test_w2v_scaled = scaler.transform(test_w2v)

    # Combine TF-IDF (sparse) with Word2Vec (dense)
    train_features = hstack([train_tfidf, csr_matrix(train_w2v_scaled)])
    test_features = hstack([test_tfidf, csr_matrix(test_w2v_scaled)])

    print(f"Combined features shape: train={train_features.shape}, test={test_features.shape}")

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train["sentiment"],
        test_size=0.2, random_state=42
    )

    # Try XGBoost
    try:
        import xgboost as xgb
        print("\nTraining XGBoost...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'nthread': 4
        }

        evals = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(params, dtrain, num_boost_round=500, evals=evals,
                          early_stopping_rounds=50, verbose_eval=50)

        y_val_proba = model.predict(dval)
        val_auc = roc_auc_score(y_val, y_val_proba)
        print(f"\nXGBoost Validation AUC: {val_auc:.4f}")

        # Predict on test
        dtest = xgb.DMatrix(test_features)
        test_predictions = (model.predict(dtest) > 0.5).astype(int)

        # Save model
        model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    except ImportError:
        print("XGBoost not available, using Logistic Regression...")
        from sklearn.linear_model import LogisticRegression

        best_auc = 0
        best_C = None
        best_model = None

        for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            lr = LogisticRegression(max_iter=3000, C=C, solver='saga', random_state=42)
            lr.fit(X_train, y_train)

            y_val_proba = lr.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_proba)
            print(f"C={C}: Validation AUC = {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_C = C
                best_model = lr

        print(f"\nBest C={best_C}, Best validation AUC: {best_auc:.4f}")
        val_auc = best_auc

        # Retrain on full data
        final_model = LogisticRegression(max_iter=3000, C=best_C, solver='saga', random_state=42)
        final_model.fit(train_features, train["sentiment"])

        test_predictions = final_model.predict(test_features)
        joblib.dump(final_model, os.path.join(MODELS_DIR, "lr_combined_model.pkl"))

    # Save submission
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_predictions})
    output.to_csv(os.path.join(SUBMISSION_DIR, "Combined_submission.csv"), index=False, quoting=3)
    print(f"\nSubmission saved to submission/Combined_submission.csv")

    return val_auc

if __name__ == "__main__":
    result = main()
    print(f"\nFinal validation AUC: {result:.4f}")
