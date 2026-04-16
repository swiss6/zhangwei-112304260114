"""
使用Google预训练Word2Vec词向量 + 均值Embedding + 逻辑回归
目标: Kaggle AUC >= 0.94
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import joblib

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")

GOOGLE_W2V_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")

# 保留否定词的停用词列表
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

def get_mean_embedding(words, model, vector_size=300):
    """Calculate mean embedding using pre-trained model"""
    feature_vec = np.zeros((vector_size,), dtype="float32")
    n_words = 0
    for word in words:
        # Google Word2Vec uses different case - try both
        if word in model:
            feature_vec += model[word]
            n_words += 1
        elif word.capitalize() in model:
            feature_vec += model[word.capitalize()]
            n_words += 1
        elif word.upper() in model:
            feature_vec += model[word.upper()]
            n_words += 1
    if n_words > 0:
        feature_vec /= n_words
    return feature_vec

def main():
    train_emb_path = os.path.join(MODELS_DIR, "train_google_embeddings.npy")
    test_emb_path = os.path.join(MODELS_DIR, "test_google_embeddings.npy")

    # Load data
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv/labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv/testData.tsv"),
                       header=0, delimiter="\t", quoting=3)

    # Check if Google Word2Vec exists
    if not os.path.exists(GOOGLE_W2V_PATH):
        print(f"\nGoogle Word2Vec model not found at: {GOOGLE_W2V_PATH}")
        print("Please download from: https://code.google.com/archive/p/word2vec/")
        print("Or run with self-trained Word2Vec instead.")
        return None

    # Load Google Word2Vec
    print("Loading Google pre-trained Word2Vec model (this may take a minute)...")
    google_model = KeyedVectors.load_word2vec_format(GOOGLE_W2V_PATH, binary=True)
    print(f"Loaded {len(google_model)} word vectors")

    # Check if embeddings already exist
    if os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        print("Loading existing embeddings...")
        train_embeddings = np.load(train_emb_path)
        test_embeddings = np.load(test_emb_path)
    else:
        # Preprocess and create embeddings
        print("Preprocessing training reviews...")
        train_reviews = []
        for i, review in enumerate(train["review"]):
            if (i+1) % 5000 == 0:
                print(f"Processing train review {i+1}/{len(train)}")
            train_reviews.append(preprocess_text(review, remove_stopwords=True))

        print("Preprocessing test reviews...")
        test_reviews = []
        for i, review in enumerate(test["review"]):
            if (i+1) % 5000 == 0:
                print(f"Processing test review {i+1}/{len(test)}")
            test_reviews.append(preprocess_text(review, remove_stopwords=True))

        # Create embeddings
        print("Creating embeddings for training data...")
        train_embeddings = np.zeros((len(train_reviews), 300), dtype="float32")
        for i, words in enumerate(train_reviews):
            train_embeddings[i] = get_mean_embedding(words, google_model)
            if (i+1) % 5000 == 0:
                print(f"  Embedded {i+1}/{len(train_reviews)}")

        print("Creating embeddings for test data...")
        test_embeddings = np.zeros((len(test_reviews), 300), dtype="float32")
        for i, words in enumerate(test_reviews):
            test_embeddings[i] = get_mean_embedding(words, google_model)
            if (i+1) % 5000 == 0:
                print(f"  Embedded {i+1}/{len(test_reviews)}")

        # Save embeddings
        print("Saving embeddings...")
        np.save(train_emb_path, train_embeddings)
        np.save(test_emb_path, test_embeddings)

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_embeddings, train["sentiment"],
        test_size=0.2, random_state=42
    )

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    best_auc = 0
    best_C = None

    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        lr = LogisticRegression(max_iter=3000, C=C, solver='lbfgs', random_state=42)
        lr.fit(X_train, y_train)

        y_val_proba = lr.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_acc = lr.score(X_val, y_val)

        print(f"C={C:8.3f}: Validation AUC = {val_auc:.4f}, Accuracy = {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_C = C

    print(f"\nBest C={best_C}, Best validation AUC: {best_auc:.4f}")

    # Retrain on full data
    print("\nRetraining on full training data...")
    final_model = LogisticRegression(max_iter=3000, C=best_C, solver='lbfgs', random_state=42)
    final_model.fit(train_embeddings, train["sentiment"])

    # Save model
    model_path = os.path.join(MODELS_DIR, "lr_google_w2v_model.pkl")
    joblib.dump(final_model, model_path)

    # Predict
    print("Making predictions...")
    test_predictions = final_model.predict(test_embeddings)

    # Save submission
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_predictions})
    output.to_csv(os.path.join(SUBMISSION_DIR, "GoogleWord2Vec_LR_submission.csv"), index=False, quoting=3)
    print(f"Submission saved to submission/GoogleWord2Vec_LR_submission.csv")

    return best_auc

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\nFinal validation AUC: {result:.4f}")
