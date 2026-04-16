"""
Bag of Words + Random Forest for Sentiment Analysis
基准模型：词袋模型 + 随机森林
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")

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

def review_to_words(raw_review):
    """Clean and tokenize text"""
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in ENGLISH_STOPWORDS]
    return " ".join(meaningful_words)

def main():
    print("Loading training data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv/labeledTrainData.tsv"),
                        header=0, delimiter="\t", quoting=3)

    print("Cleaning and parsing training set movie reviews...\n")
    num_reviews = len(train["review"])
    clean_train_reviews = []

    for i in range(0, num_reviews):
        if (i+1) % 5000 == 0:
            print("Review %d of %d" % (i+1, num_reviews))
        clean_train_reviews.append(review_to_words(train["review"][i]))

    print("Creating the bag of words...\n")
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                                 stop_words=None, max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    print("Training the random forest...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])

    print("Loading test data...")
    test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv/testData.tsv"),
                       header=0, delimiter="\t", quoting=3)

    print("Cleaning and parsing test set movie reviews...\n")
    num_test_reviews = len(test["review"])
    clean_test_reviews = []

    for i in range(0, num_test_reviews):
        if (i+1) % 5000 == 0:
            print("Review %d of %d" % (i+1, num_test_reviews))
        clean_test_reviews.append(review_to_words(test["review"][i]))

    print("Creating bag of words for test data...")
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    print("Making predictions on test data...")
    result = forest.predict(test_data_features)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(os.path.join(SUBMISSION_DIR, "Bag_of_Words_model.csv"),
                  index=False, quoting=3)

    print("Done! Submission file created: submission/Bag_of_Words_model.csv")

if __name__ == "__main__":
    main()
