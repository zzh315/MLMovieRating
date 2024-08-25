
# Movie Rating Prediction 

-   **Author:** Zhonghe Zheng	
-   **Student No:** 825612
## Project Overview

This project aims to predict movie ratings based on features extracted from the TMDB dataset using various machine learning models.  

-   **Feature Engineering:** Seperate and combine features from text and numerical data to investigate features that affect accuracy.
-   **Model Evaluation:** Testing different models to analyze their ability to predict movie ratings.
-   **Result Interpretation:** Understanding model performance via precision, recall, and accuracy metrics.

## Data Description

### 1. **Datasets:**

-   **TMDB_train.csv:**
    
    -   Training dataset containing 100,000 movie records with labels.
    -   Includes 44 columns with information like `title`, `release_year`, `runtime`, and text features.
-   **TMDB_eval.csv:**
    
    -   Evaluation dataset containing 20,000 movie records.
    -   Features and labels are similar to the training dataset.
-   **TMDB_test.csv:**
    
    -   Test dataset containing 20,000 movie records for final testing.
    -   Features are the same as the training set, excluding labels.

### 2. **Text Feature Engineering:**

-   **Bag-of-Words (BoW):** Converts textual features into numeric form using a token count matrix.
-   **TF-IDF:** Assigns a weight to each token based on its importance in the document.

## Notebooks and Code

### A2.ipynb

The main notebook contains the following steps:

1.  **Data Loading and Preprocessing:**
    
    -   Loads and preprocesses the training, evaluation, and test datasets.
    -   Handles missing data, normalizes numerical features, and vectorizes text features using **BoW** and **TF-IDF**.
2.  **Model Implementation and Training:**
    
    -   Implements multiple machine learning models, includes:
        -   **Multinomial Naive Bayes**
        -   **Decision Tree Classifier**
        -   **Logistic Regression**
        -   **Multi-Layer Perceptron**
        -   **K-Nearest Neighbors**
    -   Trains each model on the training dataset and evaluates on the evaluation dataset.
3.  **Evaluation and Analysis:**
    
    -   Measures performance using precision, recall, and accuracy scores.
    -   Compares models to understand the strengths and weaknesses of each approach.
    -   Analyzes the importance of features and combination of them.
4.  **Prediction on Test Set:**
    
    -   Applies the final selected models to predict labels for the test set.
    -   Prepares the results for submission to Kaggle or further analysis.

## Usage Instructions

1.  **Environment Setup:**
    
    -   Ensure Python (3.8.0^) and key libraries like `scikit-learn`, `pandas`, and `numpy` are installed.
    -   Use `pip` or `conda` to install dependencies:
    
    `pip install pandas `   
    `pip install numpy `   
    `pip install matplotlib `   
    `pip install scipy `   
    `pip install sklearn ` 
    
2.  **Running the Notebook:**
    
    -   Open the `A2.ipynb` notebook using VSCode or another Python notebook tool.
    -   Execute each code cell sequentially to preprocess data, train models, and evaluate results.

## Results 

The notebook provides analytical results of model performance and insights into which models are best suited for predicting movie ratings based on the TMDB dataset.
