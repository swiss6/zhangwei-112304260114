"""
Doc2Vec + Logistic Regression for Sentiment Analysis
Doc2Vec方法：直接学习文档向量
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ENGLISH_STOPWORDS = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'could',
    "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in',
    'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no',
    'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that',
    "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
    "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
    "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you',
    "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

def preprocess_text(raw_review, remove_stopwords=True):
    """Clean and tokenize text"""
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in ENGLISH_STOPWORDS]
    return words

def main():
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv/labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv/testData.tsv"),
                       header=0, delimiter="\t", quoting=3)

    print("Preprocessing all reviews...")
    all_reviews = []
    all_reviews.extend([preprocess_text(r, remove_stopwords=True) for r in train["review"]])
    all_reviews.extend([preprocess_text(r, remove_stopwords=True) for r in test["review"]])

    print("Creating tagged documents for Doc2Vec...")
    tagged_docs = [TaggedDocument(words=review, tags=[str(i)]) for i, review in enumerate(all_reviews)]

    print("Training Doc2Vec model...")
    model = Doc2Vec(documents=tagged_docs, vector_size=300, window=5, min_count=2, workers=4, epochs=40, dm=1)

    print("Saving Doc2Vec model...")
    model.save(os.path.join(MODELS_DIR, "doc2vec.model"))

    print("Creating document embeddings...")
    train_tagged = [TaggedDocument(words=preprocess_text(r, remove_stopwords=True), tags=[str(i)]) for i, r in enumerate(train["review"])]
    train_embeddings = np.array([model.infer_vector(doc.words) for doc in train_tagged])

    print("Saving train embeddings...")
    np.save(os.path.join(MODELS_DIR, "train_embeddings.npy"), train_embeddings)

    X_train, X_val, y_train, y_val = train_test_split(train_embeddings, train["sentiment"], test_size=0.2, random_state=42)

    print("Training Logistic Regression with different C values...")
    best_acc = 0
    best_model = None
    for C in [0.5, 1.0, 2.0, 5.0, 10.0]:
        lr = LogisticRegression(max_iter=1000, C=C, solver='lbfgs')
        lr.fit(X_train, y_train)
        val_acc = lr.score(X_val, y_val)
        print(f"C={C}: Validation accuracy = {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = lr

    print(f"\nBest validation accuracy: {best_acc:.4f}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))

    print("Done!")

if __name__ == "__main__":
    main()
