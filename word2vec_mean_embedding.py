"""
Word2Vec + Mean Embedding + Logistic Regression for Sentiment Analysis
目标: ROC AUC >= 0.94
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import joblib

# English stopwords
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

DATA_DIR = "D:/Code/pythonDemo/Bag of Words Meets Bags of Popcorn"

def preprocess_text(raw_review, remove_stopwords=True):
    """Clean and tokenize text"""
    # Remove HTML
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # Convert to lowercase and split
    words = letters_only.lower().split()
    # Remove stopwords
    if remove_stopwords:
        words = [w for w in words if w not in ENGLISH_STOPWORDS]
    return words

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
    # Check if Word2Vec model already exists
    w2v_model_path = os.path.join(DATA_DIR, "word2vec.model")
    train_emb_path = os.path.join(DATA_DIR, "train_w2v_embeddings.npy")
    test_emb_path = os.path.join(DATA_DIR, "test_w2v_embeddings.npy")

    # Load data
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv/labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv/testData.tsv"),
                       header=0, delimiter="\t", quoting=3)

    # Check if we need to train Word2Vec
    if os.path.exists(w2v_model_path) and os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        print("Loading existing Word2Vec model and embeddings...")
        w2v_model = Word2Vec.load(w2v_model_path)
        train_embeddings = np.load(train_emb_path)
        test_embeddings = np.load(test_emb_path)
        print(f"Loaded embeddings: train {train_embeddings.shape}, test {test_embeddings.shape}")
    else:
        print("Preprocessing all reviews...")
        # Preprocess training reviews
        train_reviews = []
        for i, review in enumerate(train["review"]):
            if (i+1) % 5000 == 0:
                print(f"Processing train review {i+1}/{len(train)}")
            train_reviews.append(preprocess_text(review, remove_stopwords=True))

        # Preprocess test reviews
        test_reviews = []
        for i, review in enumerate(test["review"]):
            if (i+1) % 5000 == 0:
                print(f"Processing test review {i+1}/{len(test)}")
            test_reviews.append(preprocess_text(review, remove_stopwords=True))

        # Combine all reviews for Word2Vec training
        all_reviews = train_reviews + test_reviews

        # Train Word2Vec model
        print("Training Word2Vec model...")
        # Parameters tuned for sentiment analysis
        w2v_model = Word2Vec(
            sentences=all_reviews,
            vector_size=300,      # Dimension of word vectors
            window=10,            # Context window size
            min_count=2,          # Minimum word frequency
            workers=4,            # Number of threads
            epochs=20,            # Number of training epochs
            sg=1,                 # Skip-gram (1) vs CBOW (0)
            sample=1e-3,          # Subsampling rate
            negative=5            # Negative sampling
        )

        # Save Word2Vec model
        print("Saving Word2Vec model...")
        w2v_model.save(w2v_model_path)

        # Create mean embeddings for training data
        print("Creating mean embeddings for training data...")
        train_embeddings = np.zeros((len(train_reviews), 300), dtype="float32")
        for i, words in enumerate(train_reviews):
            train_embeddings[i] = get_mean_embedding(words, w2v_model, 300)
            if (i+1) % 5000 == 0:
                print(f"  Embedded {i+1}/{len(train_reviews)} train reviews")

        # Create mean embeddings for test data
        print("Creating mean embeddings for test data...")
        test_embeddings = np.zeros((len(test_reviews), 300), dtype="float32")
        for i, words in enumerate(test_reviews):
            test_embeddings[i] = get_mean_embedding(words, w2v_model, 300)
            if (i+1) % 5000 == 0:
                print(f"  Embedded {i+1}/{len(test_reviews)} test reviews")

        # Save embeddings
        print("Saving embeddings...")
        np.save(train_emb_path, train_embeddings)
        np.save(test_emb_path, test_embeddings)

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_embeddings, train["sentiment"],
        test_size=0.2, random_state=42
    )

    # Train Logistic Regression with hyperparameter tuning
    print("\nTraining Logistic Regression with hyperparameter tuning...")
    best_auc = 0
    best_model = None
    best_C = None

    for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        lr = LogisticRegression(max_iter=2000, C=C, solver='lbfgs', random_state=42)
        lr.fit(X_train, y_train)

        # Get probabilities for ROC AUC
        y_val_proba = lr.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_acc = lr.score(X_val, y_val)

        print(f"C={C:6.2f}: Validation AUC = {val_auc:.4f}, Accuracy = {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = lr
            best_C = C

    print(f"\nBest C={best_C}, Best validation AUC: {best_auc:.4f}")

    # Save the best model
    model_path = os.path.join(DATA_DIR, "lr_w2v_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    # Retrain on full training data with best C
    print("\nRetraining on full training data...")
    final_model = LogisticRegression(max_iter=2000, C=best_C, solver='lbfgs', random_state=42)
    final_model.fit(train_embeddings, train["sentiment"])

    # Make predictions on test data
    print("Making predictions on test data...")
    test_predictions = final_model.predict(test_embeddings)

    # Create submission file
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_predictions})
    submission_path = os.path.join(DATA_DIR, "Word2Vec_LR_submission.csv")
    output.to_csv(submission_path, index=False, quoting=3)
    print(f"Submission file saved to {submission_path}")

    print("\nDone!")
    return best_auc

if __name__ == "__main__":
    best_auc = main()
    print(f"\nFinal validation AUC: {best_auc:.4f}")
