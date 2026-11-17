
import os
import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Download NLTK resources
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {
    "not", "no", "nor", "never", "n't"
}

os.makedirs("outputs", exist_ok=True)


# 1. Load training and testing datasets
def load_data():

    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test.csv")

    required_columns = {"review_text", "label"}

    for name, df in [("train", train_df), ("test", test_df)]:
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"{name}.csv must contain review_text and label columns"
            )

    train_df = train_df.dropna(
        subset=["review_text", "label"]
    ).copy()

    test_df = test_df.dropna(
        subset=["review_text", "label"]
    ).copy()

    for df in [train_df, test_df]:
        df["review_text"] = df["review_text"].astype(str)
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    valid_labels = {"positive", "negative"}

    train_df = train_df[
        train_df["label"].isin(valid_labels)
    ].reset_index(drop=True)

    test_df = test_df[
        test_df["label"].isin(valid_labels)
    ].reset_index(drop=True)

    if train_df["label"].nunique() != 2:
        raise ValueError("Training data must contain both classes.")

    if test_df.empty:
        raise ValueError("Testing dataset is empty.")

    print("\n===== DATASET INFORMATION =====")
    print("Training reviews:", len(train_df))
    print("Testing reviews:", len(test_df))

    print("\nTraining distribution:")
    print(train_df["label"].value_counts())

    print("\nTesting distribution:")
    print(test_df["label"].value_counts())

    return train_df, test_df


# 2. Text preprocessing
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)

    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


# 3. VADER sentiment analysis
def evaluate_vader(test_df):

    sia = SentimentIntensityAnalyzer()

    actual = test_df["label"].tolist()

    predictions = []

    for review in test_df["review_text"]:

        score = sia.polarity_scores(review)["compound"]

        if score >= 0.05:
            predictions.append("positive")
        elif score <= -0.05:
            predictions.append("negative")
        else:
            predictions.append("neutral")

    print("\n===== VADER EVALUATION =====")

    print(
        classification_report(
            actual,
            predictions,
            labels=["negative", "neutral", "positive"],
            zero_division=0
        )
    )

    accuracy = accuracy_score(actual, predictions)

    print("VADER Accuracy:", round(accuracy, 4))

    return predictions


# 4. Train TF-IDF + Logistic Regression
def train_model(train_df, test_df):

    X_train = train_df["review_text"].apply(clean_text)
    y_train = train_df["label"]

    X_test = test_df["review_text"].apply(clean_text)
    y_test = test_df["label"]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("\nTraining Logistic Regression model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=["negative", "positive"],
        zero_division=0
    )

    print("\n===== LOGISTIC REGRESSION RESULTS =====")

    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(report)

    # Save model
    joblib.dump(model, "outputs/sentiment_model.pkl")

    # Save evaluation report
    with open("outputs/model_evaluation.txt", "w") as file:
        file.write("Logistic Regression Evaluation\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "positive"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    display.plot(cmap="Blues")

    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    return model, predictions


# 5. Display important words
def top_words(model, n=10):

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    coefficients = classifier.coef_[0]
    words = vectorizer.get_feature_names_out()

    ranked = sorted(zip(coefficients, words))

    print("\n===== IMPORTANT WORDS =====")

    print("Negative words:")
    print([word for _, word in ranked[:n]])

    print("\nPositive words:")
    print([word for _, word in ranked[-n:][::-1]])


# 6. Sentiment distribution
def plot_distribution(test_df, predictions):

    results = pd.DataFrame({
        "Actual": test_df["label"].values,
        "Predicted": predictions
    })

    results["Predicted"].value_counts().plot(kind="bar")

    plt.title("Predicted Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("outputs/sentiment_distribution.png", dpi=150)
    plt.close()


# 7. Predict new reviews
def predict(model, reviews):

    cleaned = [clean_text(review) for review in reviews]

    probabilities = model.predict_proba(cleaned)

    classes = model.named_steps["classifier"].classes_

    positive_index = list(classes).index("positive")

    print("\n===== NEW REVIEW PREDICTIONS =====")

    for review, probability in zip(reviews, probabilities):

        positive_probability = probability[positive_index]

        sentiment = (
            "Positive"
            if positive_probability >= 0.5
            else "Negative"
        )

        print(
            f"[{sentiment} | Positive probability: "
            f"{positive_probability:.2f}] {review}"
        )


# 8. Main program
if __name__ == "__main__":

    train_df, test_df = load_data()

    # VADER
    vader_predictions = evaluate_vader(test_df)

    # Machine Learning
    model, ml_predictions = train_model(train_df, test_df)

    # Important words
    top_words(model)

    # Visualization
    plot_distribution(test_df, ml_predictions)

    # Predictions on new reviews
    predict(model, [
        "This is an excellent product. I absolutely love it!",
        "Terrible quality. Completely disappointed.",
        "The product is okay but could be better."
    ])

    print("\nAll tasks completed successfully!")
    print("Model saved in outputs/sentiment_model.pkl")
