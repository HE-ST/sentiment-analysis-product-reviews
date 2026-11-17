# Sentiment Analysis on Product Reviews

A Natural Language Processing (NLP) and Machine Learning project that analyzes Amazon product reviews and classifies customer opinions as positive or negative.

This project implements and compares two sentiment analysis approaches:

1. **VADER** – Rule-based sentiment analysis using a predefined sentiment lexicon.
2. **TF-IDF + Logistic Regression** – A supervised machine learning model trained on labeled product reviews.

The objective is to compare traditional lexicon-based sentiment analysis with a machine learning approach.

## Project Overview

| Component | Description |
|---|---|
| Domain | Natural Language Processing |
| Problem | Sentiment Classification |
| Dataset | Amazon Product Reviews |
| Training Samples | 5,000 |
| Testing Samples | 1,000 |
| ML Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |
| Baseline | VADER |

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Joblib
- TF-IDF Vectorization
- Logistic Regression

## Project Architecture

```text
Amazon Product Reviews
          |
          v
    Data Loading
          |
          v
   Text Preprocessing
          |
          v
    Sentiment Analysis
          |
     +----+----+
     |         |
     v         v
   VADER    TF-IDF
     |         |
     |         v
     |    Logistic Regression
     |         |
     +----+----+
          |
          v
    Model Evaluation
          |
          v
 Accuracy, Precision,
 Recall and F1-score
          |
          v
   Prediction on New Reviews
```

## Dataset

Source: [Cleanlab Amazon Reviews Dataset](https://huggingface.co/datasets/Cleanlab/amazon-reviews)

The dataset contains 6,000 product reviews with binary sentiment labels.

| Dataset Split | Number of Reviews |
|---|---:|
| Training | 5,000 |
| Testing | 1,000 |
| Total | 6,000 |

Dataset columns:

- `review_text`: Customer product review.
- `label`: Positive or negative sentiment.

### Class Distribution

**Training dataset**
- Positive: 3,092
- Negative: 1,908

**Testing dataset**
- Positive: 553
- Negative: 447

See `dataset/README.md` for download instructions.

## Methodology

### 1. Text Preprocessing

The reviews undergo the following preprocessing operations:

- Convert text to lowercase.
- Remove URLs.
- Remove HTML tags.
- Remove punctuation and unwanted characters.
- Remove English stopwords while preserving important negation words.

### 2. VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool.

It calculates a compound sentiment score.

Classification thresholds:

- Compound score >= 0.05: Positive
- Compound score <= -0.05: Negative
- Otherwise: Neutral

VADER does not require model training.

### 3. TF-IDF + Logistic Regression

The machine learning pipeline consists of:

**TF-IDF Vectorization**

Converts textual reviews into numerical feature vectors based on word importance.

Configuration:
- N-gram range: (1, 2)
- Minimum document frequency: 2
- Maximum features: 50,000
- Sublinear TF: Enabled

**Logistic Regression**

A supervised classification algorithm that predicts whether a review expresses positive or negative sentiment.

Configuration:
- Maximum iterations: 1,000
- Class weight: Balanced
- Random state: 42

## Experimental Results

Both approaches were evaluated using the same independent test dataset of 1,000 reviews.

### Accuracy Comparison

| Model | Accuracy |
|---|---:|
| VADER | 74.60% |
| TF-IDF + Logistic Regression | 95.10% |

### Logistic Regression Classification Report

| Metric | Negative | Positive |
|---|---:|---:|
| Precision | 0.99 | 0.93 |
| Recall | 0.90 | 0.99 |
| F1-score | 0.94 | 0.96 |

Overall:
- Accuracy: 95.10%
- Macro F1-score: 95%
- Weighted F1-score: 95%

### Results Interpretation

Logistic Regression achieved 95.10% accuracy, outperforming VADER by 20.50 percentage points.

The results indicate that the supervised machine learning approach performs better than the rule-based baseline on this dataset.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The confusion matrix is generated and saved as an image.

## Example Predictions

The trained model was tested on new product reviews.

| Review | Predicted Sentiment | Positive Probability |
|---|---|---:|
| Excellent product. I absolutely love it! | Positive | 0.93 |
| Terrible quality. Completely disappointed. | Negative | 0.12 |
| The product is okay but could be better. | Negative | 0.26 |

Note: This is a binary classification model. It does not have a separate neutral prediction class.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-product-reviews.git
```

### 2. Navigate to the project directory

```bash
cd sentiment-analysis-product-reviews
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the dataset

Download `train.csv` and `test.csv` from:

https://huggingface.co/datasets/Cleanlab/amazon-reviews

Place both files inside the `dataset/` directory.

### 7. Run the project

```bash
python sentiment_analysis.py
```

The script trains the model, evaluates both approaches and saves the trained model and evaluation outputs.

## Output Files

```text
outputs/
├── sentiment_model.pkl
├── confusion_matrix.png
├── sentiment_distribution.png
└── model_evaluation.txt
```

## Future Improvements

- Experiment with Naive Bayes and Support Vector Machines.
- Perform hyperparameter tuning.
- Apply cross-validation.
- Explore advanced NLP models such as BERT.
- Improve handling of sarcasm and contextual sentiment.
- Extend the model to support additional sentiment categories.

## Limitations

- The model uses binary sentiment classification.
- Performance depends on the quality and distribution of training data.
- TF-IDF does not fully understand context or sarcasm.
- The reported evaluation scores apply to this particular dataset and test split.

## Author

Developed as an NLP and Machine Learning project.

## Acknowledgements

- [Cleanlab Amazon Reviews Dataset](https://huggingface.co/datasets/Cleanlab/amazon-reviews)
- NLTK
- Scikit-learn

## License

The source code in this repository is released under the MIT License.

The dataset has its own MIT license and attribution information. Refer to its original Hugging Face repository for details.
