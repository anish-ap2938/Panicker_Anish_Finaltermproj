# Drinking Behavior Classification using RF, NB, LSTM.

This repository contains the code and documentation for a final project that implements three different classification algorithms to predict an individual’s drinking behavior (drinker vs. non-drinker) using health and behavioral data. The three algorithms employed are:

- **Random Forest** (ensemble learning)
- **Naive Bayes** (Gaussian variant)
- **Long Short-Term Memory (LSTM)** (deep learning)

The performance of each model is evaluated using 10-fold cross-validation, with manual calculation of various performance metrics derived from the confusion matrix.

---

## Overview

The primary objective of this project is to compare the classification performance of three distinct algorithms on a real-world dataset. The dataset comprises various demographic, health, and behavioral indicators, with the target variable `DRK_YN` indicating whether an individual is a drinker (`Y`) or non-drinker (`N`). 

The project follows these key steps:
- **Data Preprocessing:**  
  - Encoding categorical variables (e.g., `sex` and `DRK_YN`)  
  - Feature selection based on demographics, health metrics, and behavioral indicators  
  - Feature scaling using StandardScaler to ensure equal contribution from all features  
- **Model Implementation:**  
  - **Random Forest:** Configured with 50 trees, maximum depth of 10, and other parameters to balance performance and prevent overfitting.
  - **Naive Bayes:** Gaussian Naive Bayes is used, with priors calculated from the training data.
  - **LSTM:** A deep learning model designed with one LSTM layer (64 units with ReLU activation), a 50% dropout layer, and a dense output layer with sigmoid activation for binary classification.
- **Evaluation:**  
  - Models are assessed using 10-fold cross-validation.
  - Performance metrics (e.g., accuracy, precision, AUC, F1 score, balanced accuracy, Brier score, etc.) are computed manually from the confusion matrix (using a library only for TP, TN, FP, FN extraction).
  - Detailed per-fold and averaged metrics are presented in tabular format for easy visualization.
- **Experimental Comparison:**  
  - The project compares the performance of the three algorithms.
  - Discussion focuses on which model performs better and why, based on evaluation metrics such as AUC, accuracy, precision, F1 score, balanced accuracy, and Brier score.

---

## Data Description

The dataset used in this project contains various health-related measurements and behavioral indicators collected by the National Health Insurance Service in Korea. The key features include:

### Demographics
- **sex:** Gender (Male/Female)
- **age:** Age in years
- **height:** Height (cm)
- **weight:** Weight (kg)
- **waistline:** Waist circumference (cm)

### Health Metrics
- **sight_left, sight_right:** Visual acuity measurements
- **hear_left, hear_right:** Hearing ability measurements
- **SBP, DBP:** Systolic and diastolic blood pressure
- **BLDS:** Blood sugar level
- **tot_chole, HDL_chole, LDL_chole, triglyceride:** Cholesterol levels
- **hemoglobin:** Blood hemoglobin levels
- **urine_protein:** Presence of protein in urine
- **serum_creatinine:** Serum creatinine levels
- **SGOT_AST, SGOT_ALT, gamma_GTP:** Liver enzyme levels

### Behavioral Indicators
- **SMK_stat_type_cd:** Smoking status code

### Target Variable
- **DRK_YN:** Drinker status (`Y` for drinker, `N` for non-drinker)

The dataset is available from Kaggle (see references below) and has been preprocessed to ensure no missing values and appropriate scaling.

---

## Implementation Details

### Data Preprocessing
- **Encoding:**  
  - `sex` is encoded as: Male = 1, Female = 0.
  - `DRK_YN` is encoded as: Y = 1, N = 0.
- **Feature Selection:**  
  - Selected features include demographics, health metrics, and the smoking status indicator.
- **Scaling:**  
  - All features are standardized using StandardScaler.
- **Preparation:**  
  - The final feature matrix (`X`) and target vector (`y`) are extracted as NumPy arrays.

### Classification Algorithms
- **Random Forest:**  
  - Parameters: `n_estimators = 50`, `max_depth = 10`, `min_samples_split = 10`, `min_samples_leaf = 5`, class weight balanced, and reproducibility ensured with `random_state = 42`.
- **Naive Bayes:**  
  - Gaussian Naive Bayes is used, with priors computed from the training data.
- **LSTM:**  
  - Architecture:  
    - LSTM layer with 64 units and ReLU activation to capture sequential patterns.
    - Dropout layer (50%) to mitigate overfitting.
    - Dense output layer with sigmoid activation for binary classification.
  - Training Parameters: 5 epochs, batch size of 16, 20% validation split, and early stopping with a patience of 2 epochs.

### Evaluation Metrics
The following metrics are computed manually using the confusion matrix (with assistance from a library for TP, TN, FP, FN extraction):
- True Positive Rate (TPR) / Sensitivity
- True Negative Rate (TNR) / Specificity
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Precision
- F1 Score
- Accuracy
- Error Rate
- Balanced Accuracy (BACC)
- True Skill Statistic (TSS)
- Heidke Skill Score (HSS)
- Brier Score

These metrics are calculated for each fold of the 10-fold cross-validation and then averaged to provide a comprehensive performance comparison.

---

## Experimental Results and Discussion

### Performance Comparison
Based on the evaluation:
- **LSTM:**  
  - Achieved the highest AUC, accuracy, precision, F1 score, and balanced accuracy, with the lowest Brier score.  
  - Its ability to capture complex sequential patterns in the data contributes to its superior performance.
- **Random Forest:**  
  - Performed well but was slightly outperformed by the LSTM in most metrics.
  - Trained faster compared to the LSTM.
- **Naive Bayes:**  
  - Showed the lowest performance, likely due to the assumption of feature independence, which is less suitable for this dataset.

### Conclusion
The experimental results indicate that the LSTM model outperforms both the Random Forest and Naive Bayes models across multiple evaluation metrics. The LSTM's strength in modeling sequential data and capturing complex interactions among features is the key factor behind its superior performance, despite a longer training time. Overall, the LSTM model is recommended for this classification task based on its consistently higher metrics.

---

## References

- **Dataset:**  
  [Smoking-Drinking Dataset on Kaggle](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset)
- **GitHub Repository:**  
  [Project Repository Link](https://github.com/anish-ap2938/Panicker_Anish_Finaltermproj)

This project is developed for educational purposes as part of a final term project in Data Mining.
