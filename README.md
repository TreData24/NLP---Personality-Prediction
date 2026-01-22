# NLP-Personality-Prediction
Personality Prediction A machine learning project predicting Neurotic vs. Non-Neurotic personality traits from social media text using NLP techniques. Includes data cleaning, feature engineering (unigrams/bigrams), and model comparison (Logistic Regression, SVM) for improved classification accuracy.


# **Personality Prediction**
**Authors:** Taylor Lewis, Kwasi Brooks\

## **Overview**
This project develops a baseline machine learning model to classify users as **Neurotic** or **Non-Neurotic** based on free-form text from social media posts. We explore NLP techniques, feature engineering, and model optimization to improve classification performance.

## **Motivation**
Textual data reveals patterns in human behavior and emotional states. Predicting personality traits like neuroticism has applications in:
*   **Targeted advertising**
*   **Mental health monitoring**

High neuroticism correlates with mental health risks (Ormel, 2013), making accurate prediction valuable.

## **Data Source**
*   **Workshop on Computational Personality Recognition: Shared Task** (Celli, 2013)\
    Publicly available, gold-labeled datasets from social media posts (e.g., **MyPersonality** and **Essays** datasets).

## **Ethical Statement**
Data was **de-identified** by original researchers. No attempts were made to identify subjects. No IRB oversight required (Benton, 2017).

## **Data Preparation**
*   Corrected CSV delimiter errors → stored as JSON
*   Combined multiple posts per user into a single text blob (delimited by `*`)
*   Removed non-English entries and irrelevant noise (case-by-case handling of URLs/emoticons)
*   Resolved multiple category assignments per user via majority label
*   Normalized text (removed punctuation/stopwords; lemmatization)

## **Initial Analysis**
*   Imbalanced classes (\~1.5× more Non-Neurotic users)
*   Posting frequency not predictive
*   Extracted **unigrams** and **bigrams** using NLTK TweetTokenizer

## **Modeling Approach**
*   **Baseline:** Logistic Regression
*   **Feature Engineering:** Unigrams, bigrams, TF-IDF (evaluated)
*   **Improved Model:** Linear SVM + Truncated SVD (TSVD) for dimensionality reduction
*   **Evaluation:** Stratified cross-validation, confusion matrix, accuracy/precision/recall

## **Results**
*   Best model: **Linear SVM with unigram/bigram counts + TSVD**
*   Performance improved over baseline, but constrained by dataset size and class imbalance

## **Limitations**
*   Small dataset (250 users) with long-tail posting distribution
*   Excluded social network metrics to maximize generalizability
*   Hardware constraints (no dedicated GPUs) → traditional ML over deep learning
*   Insufficient data to assess bias → **Do not deploy without bias analysis**

## **How to Run**
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure parser arguments** to:
    *   Set input file
    *   Toggle removal of stopwords/punctuation
    *   Toggle stratification
    *   Set CV folds
    *   Set test set size
3.  **Change classifier**:
    *   Uncomment lines **337–345** in `main.py`
4.  **Enable TSVD**:
    *   Uncomment lines **314–334**

## **References**

*   Benton, A., Coppersmith, G., & Dredze, M. (2017). *Ethical Research Protocols for Social Media Health Research.*
*   Celli, F., Panesi, F., Stillwell, D., & Kosinski, M. (2013). *Workshop on Computational Personality Recognition: Shared Task.*
*   Dey, S., Duff, B., Chhaya, N., Fu, W., Swaminathan, V., Karahalios, K. (2018). *Recommendation for Video Advertisements based on Personality Traits.*
*   Ormel, J., Jeronimus, B. F., Kotov, R., et al. (2013). *Neuroticism and common mental disorders.*
