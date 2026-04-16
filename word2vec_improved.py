"""
Improved Word2Vec + TF-IDF Weighted Embedding + Logistic Regression
优化目标: Kaggle AUC >= 0.94
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
import joblib

# 保留否定词和情感相关词，只移除真正无意义的停用词
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

# 否定词列表 - 这些词对情感分析很重要
NEGATION_WORDS = set([
    'no', 'not', 'nor', "n't", 'never', 'nobody', 'nothing', 'nowhere', 'neither',
    'none', "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
    "can't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"
])

DATA_DIR = "D:/Code/pythonDemo/Bag of Words Meets Bags of Popcorn"

def preprocess_text_improved(raw_review, remove_stopwords=True):
    """Improved text preprocessing"""
    # Remove HTML
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()

    # Handle contractions and negations
    review_text = review_text.lower()
    review_text = re.sub(r"n't", " not", review_text)
    review_text = re.sub(r"'re", " are", review_text)
    review_text = re.sub(r"'s", " is", review_text)
    review_text = re.sub(r"'d", " would", review_text)
    review_text = re.sub(r"'ll", " will", review_text)
    review_text = re.sub(r"'ve", " have", review_text)
    review_text = re.sub(r"'m", " am", review_text)

    # Remove non-letters but keep important punctuation patterns
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # Split into words
    words = letters_only.split()

    # Remove stopwords but keep negation words
    if remove_stopwords:
        words = [w for w in words if w not in MINIMAL_STOPWORDS or w in NEGATION_WORDS]

    return words

def get_tfidf_weighted_embedding(words, model, tfidf_dict, vector_size, default_tfidf=0.001):
    """Calculate TF-IDF weighted embedding for a document"""
    feature_vec = np.zeros((vector_size,), dtype="float32")
    total_weight = 0.0

    for word in words:
        if word in model.wv:
            tfidf_weight = tfidf_dict.get(word, default_tfidf)
            feature_vec += model.wv[word] * tfidf_weight
            total_weight += tfidf_weight

    if total_weight > 0:
        feature_vec /= total_weight

    return feature_vec

def get_mean_embedding(words, model, vector_size):
    """Calculate mean embedding for a document"""
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
    w2v_model_path = os.path.join(DATA_DIR, "word2vec_improved.model")
    train_emb_path = os.path.join(DATA_DIR, "train_w2v_embeddings_improved.npy")
    test_emb_path = os.path.join(DATA_DIR, "test_w2v_embeddings_improved.npy")
    tfidf_path = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")

    # Load data
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv/labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv/testData.tsv"),
                       header=0, delimiter="\t", quoting=3)

    # Check if we need to recompute
    if os.path.exists(w2v_model_path) and os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        print("Loading existing model and embeddings...")
        w2v_model = Word2Vec.load(w2v_model_path)
        train_embeddings = np.load(train_emb_path)
        test_embeddings = np.load(test_emb_path)
        tfidf_vectorizer = joblib.load(tfidf_path)
        print(f"Loaded embeddings: train {train_embeddings.shape}, test {test_embeddings.shape}")
    else:
        print("Preprocessing all reviews...")
        train_reviews = []
        for i, review in enumerate(train["review"]):
            if (i+1) % 5000 == 0:
                print(f"Processing train review {i+1}/{len(train)}")
            train_reviews.append(preprocess_text_improved(review, remove_stopwords=True))

        test_reviews = []
        for i, review in enumerate(test["review"]):
            if (i+1) % 5000 == 0:
                print(f"Processing test review {i+1}/{len(test)}")
            test_reviews.append(preprocess_text_improved(review, remove_stopwords=True))

        all_reviews = train_reviews + test_reviews

        # Train Word2Vec with improved parameters
        print("Training Word2Vec model...")
        w2v_model = Word2Vec(
            sentences=all_reviews,
            vector_size=300,
            window=10,
            min_count=2,
            workers=4,
            epochs=30,           # More epochs
            sg=1,
            sample=1e-3,
            negative=10,         # More negative samples
            alpha=0.025,
            min_alpha=0.0001
        )

        print("Saving Word2Vec model...")
        w2v_model.save(w2v_model_path)

        # Build TF-IDF for weighting
        print("Building TF-IDF weights...")
        train_texts = [' '.join(words) for words in train_reviews]
        test_texts = [' '.join(words) for words in test_reviews]

        tfidf_vectorizer = TfidfVectorizer(max_features=50000, min_df=2, max_df=0.95)
        tfidf_vectorizer.fit(train_texts + test_texts)

        # Create word -> TF-IDF weight mapping
        tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(),
                              tfidf_vectorizer.idf_))
        joblib.dump(tfidf_vectorizer, tfidf_path)

        # Create TF-IDF weighted embeddings for training data
        print("Creating TF-IDF weighted embeddings for training data...")
        train_embeddings = np.zeros((len(train_reviews), 300), dtype="float32")
        for i, words in enumerate(train_reviews):
            train_embeddings[i] = get_tfidf_weighted_embedding(words, w2v_model, tfidf_dict, 300)
            if (i+1) % 5000 == 0:
                print(f"  Embedded {i+1}/{len(train_reviews)} train reviews")

        # Create TF-IDF weighted embeddings for test data
        print("Creating TF-IDF weighted embeddings for test data...")
        test_embeddings = np.zeros((len(test_reviews), 300), dtype="float32")
        for i, words in enumerate(test_reviews):
            test_embeddings[i] = get_tfidf_weighted_embedding(words, w2v_model, tfidf_dict, 300)
            if (i+1) % 5000 == 0:
                print(f"  Embedded {i+1}/{len(test_reviews)} test reviews")

        print("Saving embeddings...")
        np.save(train_emb_path, train_embeddings)
        np.save(test_emb_path, test_embeddings)

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_embeddings, train["sentiment"],
        test_size=0.2, random_state=42
    )

    # Train Logistic Regression with extensive hyperparameter tuning
    print("\nTraining Logistic Regression with hyperparameter tuning...")
    best_auc = 0
    best_model = None
    best_C = None

    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        lr = LogisticRegression(max_iter=3000, C=C, solver='lbfgs', random_state=42)
        lr.fit(X_train, y_train)

        y_val_proba = lr.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_acc = lr.score(X_val, y_val)

        print(f"C={C:8.3f}: Validation AUC = {val_auc:.4f}, Accuracy = {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = lr
            best_C = C

    print(f"\nBest C={best_C}, Best validation AUC: {best_auc:.4f}")

    # Save the best model
    model_path = os.path.join(DATA_DIR, "lr_w2v_model_improved.pkl")
    joblib.dump(best_model, model_path)

    # Retrain on full training data
    print("\nRetraining on full training data...")
    final_model = LogisticRegression(max_iter=3000, C=best_C, solver='lbfgs', random_state=42)
    final_model.fit(train_embeddings, train["sentiment"])

    # Make predictions
    print("Making predictions on test data...")
    test_predictions = final_model.predict(test_embeddings)

    # Create submission file
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_predictions})
    submission_path = os.path.join(DATA_DIR, "Word2Vec_LR_submission_improved.csv")
    output.to_csv(submission_path, index=False, quoting=3)
    print(f"Submission file saved to {submission_path}")

    print("\nDone!")
    return best_auc

if __name__ == "__main__":
    best_auc = main()
    print(f"\nFinal validation AUC: {best_auc:.4f}")
