# Spam Email Classifier

This project presents a comprehensive solution for building a **Spam Email Classifier** using Python and a robust machine learning pipeline. The core objective is to accurately distinguish between legitimate (ham) and unsolicited (spam) emails by leveraging natural language processing (NLP) techniques and a supervised learning model. The classifier is trained on a real-world dataset of emails, demonstrating a practical approach to a common machine learning problem.

---

## Project Overview and Methodology

The project's methodology is structured as a multi-stage process, meticulously designed to transform raw, unstructured email data into a format suitable for machine learning.

### 1. Data Acquisition and Preparation

The foundation of any machine learning project is a high-quality dataset. For this project, we utilize the **Apache SpamAssassin public corpus**, a widely recognized and realistic collection of spam and ham emails. The process begins with programmatically downloading and extracting these compressed datasets.

- **Email Parsing**: Raw emails are not directly usable. They are often complex, multipart structures containing headers, various content types (e.g., plain text, HTML), and attachments. Python's built-in `email` module is employed to parse these intricate structures, enabling us to systematically extract the relevant text content. This step includes a crucial sub-task of converting HTML content into clean, readable plain text, ensuring a consistent representation across all emails.

### 2. Natural Language Processing (NLP)

Once the raw text content is extracted, a series of NLP techniques are applied to clean and standardize the data. This preprocessing is vital for the model's performance, as it reduces noise and focuses on the most informative features.

- **Text Preprocessing**: The text undergoes several cleaning steps:
    - **Lowercasing**: All text is converted to lowercase to treat words like "Free" and "free" as identical, reducing the vocabulary size.
    - **URL Replacement**: URLs, while common in emails, often don't contribute meaningfully to classification and can introduce noise. They are replaced with a standardized placeholder, "URL", using a library like `urlextract`.
    - **Number Replacement**: Similarly, sequences of digits are replaced with a placeholder, "NUMBER", to generalize the model and prevent it from overfitting to specific numerical values (e.g., zip codes, phone numbers).
    - **Punctuation Removal**: Punctuation marks are removed to focus on the semantic content of the words themselves.
    - **Stemming**: We use stemming algorithms (e.g., from the `nltk` library) to reduce words to their root form. For example, "running," "runs," and "ran" are all reduced to "run." This process helps in conflating words with similar meanings, further reducing the vocabulary and improving generalization.

### 3. Feature Engineering and Vectorization

Machine learning models require numerical input. This stage transforms the cleaned text into a numerical format.

- **Tokenization and Vocabulary Creation**: The preprocessed text is tokenized into individual words. A vocabulary of the most frequent words is then built from the training data. This controlled vocabulary ensures that the feature space remains manageable and relevant.
- **Word Count Vectorization**: Each email is converted into a vector where each element represents the count of a specific word from our vocabulary. This results in a Bag-of-Words representation.
- **Sparse Matrix Representation**: Given that most emails only contain a small subset of the total vocabulary, the resulting feature vectors are often sparse (i.e., contain many zeros). To handle this efficiently, the data is stored in a sparse matrix format (`scipy.sparse`), which significantly reduces memory usage and improves computational performance.
- **`scikit-learn` Pipelines**: To ensure a clean and reproducible workflow, a `scikit-learn` `Pipeline` is constructed. This powerful tool chains the custom preprocessing steps (implemented as `scikit-learn`-compatible transformers) with the vectorization and final classification model. This prevents data leakage and simplifies the entire process.

### 4. Model Training and Evaluation

A machine learning model is trained on the vectorized data to learn the patterns that differentiate spam from ham.

- **Logistic Regression**: A **Logistic Regression** classifier is chosen for its simplicity, efficiency, and interpretability. It is a powerful linear model that is well-suited for high-dimensional, sparse datasets like the ones generated from text.
- **Cross-Validation**: To ensure the model's robustness and to prevent overfitting, a stratified K-fold cross-validation approach is employed during training. This provides a more reliable estimate of the model's performance on unseen data.
- **Performance Metrics**: The final model's performance is rigorously evaluated on a held-out test set that was not used during training. We focus on key metrics that are particularly relevant for imbalanced classification problems like spam detection:
    - **Precision**: The proportion of correctly identified spam emails out of all emails classified as spam. High precision is crucial to minimize the number of legitimate emails incorrectly flagged as spam (false positives).
    - **Recall**: The proportion of correctly identified spam emails out of all actual spam emails. High recall is important to ensure that most spam emails are caught and do not reach the user's inbox (false negatives).

---

## Technical Skills Demonstrated

- **Data Engineering**: Proficiently handling data from external sources, including downloading, unpacking, and parsing complex file formats.
- **Email Processing**: Deep understanding and practical application of Python's `email` module for parsing multipart emails and extracting meaningful text.
- **Natural Language Processing (NLP)**: Expertise in text cleaning, normalization, tokenization, and stemming, including the use of regular expressions (`re`) for pattern matching.
- **Feature Engineering**: Designing and implementing custom `scikit-learn` transformers for creating scalable, reusable preprocessing steps. Creating sparse feature matrices for efficient handling of high-dimensional data.
- **Machine Learning**: End-to-end model development lifecycle, from data preprocessing to model training and rigorous evaluation. Experience with `scikit-learn`'s `Pipeline` API for building robust and reproducible workflows.
- **Python Libraries**: Comprehensive command of essential scientific computing libraries, including `numpy` for numerical operations, `scipy` for sparse matrices, and `scikit-learn` for machine learning.
- **Software Engineering**: Writing modular and maintainable code, including graceful handling of missing dependencies (e.g., `urlextract`).

---

## How to Use

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/spam-email-classifier.git](https://github.com/your-username/spam-email-classifier.git)
    cd spam-email-classifier
    ```
2.  **Install Dependencies**: Ensure you have all the necessary libraries by running:
    ```bash
    pip install numpy scipy scikit-learn urlextract
    ```
    (Note: `urlextract` is used for URL replacement. If you prefer, you can modify the code to use regular expressions instead.)
3.  **Run the Main Script**:
    ```bash
    python spam_classifier.py
    ```
    The script will automatically download the dataset, train the model, and print the evaluation metrics to the console.
