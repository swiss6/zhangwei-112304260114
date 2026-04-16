import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

def preprocess_text(raw_review):
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return " ".join(words)

DATA_DIR = "d:/Code/pythonDemo/Bag of Words Meets Bags of Popcorn"

print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/labeledTrainData.tsv/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
test = pd.read_csv(f"{DATA_DIR}/testData.tsv/testData.tsv",
                   header=0, delimiter="\t", quoting=3)

print("Preprocessing reviews...")
clean_train = [preprocess_text(r) for r in train["review"]]
clean_test = [preprocess_text(r) for r in test["review"]]

print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_train = tfidf.fit_transform(clean_train)
X_test = tfidf.transform(clean_test)

print("Saving TF-IDF features...")
joblib.dump(tfidf, f"{DATA_DIR}/tfidf_vectorizer.pkl")
np.save(f"{DATA_DIR}/train_tfidf_features.npy", X_train.toarray())
np.save(f"{DATA_DIR}/test_tfidf_features.npy", X_test.toarray())

print("\nTraining Logistic Regression on full training data...")
lr = LogisticRegression(max_iter=3000, C=8.0, solver='lbfgs')
lr.fit(X_train, train["sentiment"])

train_acc = lr.score(X_train, train["sentiment"])
print(f"Training accuracy: {train_acc:.4f}")

joblib.dump(lr, f"{DATA_DIR}/best_model.pkl")

print("\nMaking predictions on test data...")
predictions = lr.predict(X_test)

output = pd.DataFrame(data={"id": test["id"], "sentiment": predictions})
output.to_csv(f"{DATA_DIR}/TF-IDF_model.csv", index=False, quoting=3)

print(f"Done! Submission file saved to TF-IDF_model.csv")