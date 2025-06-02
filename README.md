# Insurance Cross-Selling Prediction Model

## Project Overview
This project develops a predictive model for AssurePredict, a leading insurance company, to identify potential cross-selling opportunities among existing customers. The goal is to predict which customers are likely to purchase vehicle insurance, enabling more targeted marketing strategies.

## Dataset
The analysis uses a dataset containing customer information including:
- Demographic data (Age, Gender)
- Vehicle information (Vehicle Age, Damage history)
- Insurance history (Previously Insured)
- Financial information (Annual Premium)
- Response variable (whether the customer purchased vehicle insurance)

## Project Structure

### 1. Exploratory Data Analysis (EDA)
- Dataset overview and missing values analysis
- Target variable distribution analysis (revealed significant class imbalance)
- Categorical and numerical feature analysis
- Statistical tests for feature associations (Chi-squared tests)
- Feature distributions and interactions
- Outlier detection using z-score method

Key Findings:
- Significant class imbalance in target variable
- Strong associations between Previously_Insured, Vehicle_Damage, and target variable
- Low correlation among numerical features (no multicollinearity issues)
- Notable outliers in Annual_Premium feature

### 2. Feature Engineering
- Categorical variable encoding:
  - Binary encoding for Gender and Vehicle_Damage
  - Ordinal encoding for Vehicle_Age
- Numerical feature scaling using StandardScaler
- Feature selection using SelectKBest

### 3. Model Development
Three different approaches were implemented to handle class imbalance:

1. Original Model (with balanced class weights):
   - Logistic Regression with class_weight='balanced'
   - ROC-AUC Score: 0.832

2. SMOTE (Synthetic Minority Over-sampling Technique):
   - Balanced dataset through synthetic sample generation
   - ROC-AUC Score: 0.786

3. Random Undersampling:
   - Balanced dataset by reducing majority class
   - ROC-AUC Score: 0.785

### 4. Feature Reduction Analysis
A reduced model was created by removing Gender_bin and Annual_Premium features:
- Maintained similar performance (ROC-AUC: 0.830)
- More interpretable with fewer features
- Avoids potential gender bias
- Requires less data collection

### 5. Threshold Optimization
Analysis of different classification thresholds (0.5 vs 0.75):
- Default threshold (0.5):
  - Precision: 0.25
  - Recall: 0.98
  - F1-score: 0.40

- Optimized threshold (0.75):
  - Precision: 0.32 (+28%)
  - Recall: 0.21 (-79%)
  - F1-score: 0.25

## Results and Recommendations
1. Model Selection:
   - The original model with balanced class weights performed best (ROC-AUC: 0.832)
   - Reduced feature set provides similar performance with better interpretability

2. Threshold Selection:
   - Default threshold (0.5): recommended if the business goal is to identify as many potential customers as possible (high recall), accepting a higher number of false positives and lower precision.
   - Higher threshold (0.75): recommended if the business goal is to focus on high-confidence leads (higher precision), accepting that many actual buyers will not be identified (lower recall). This is suitable when the cost of false positives is high or resources for follow-up are limited.

3. Key Predictive Features:
   - Previously_Insured
   - Vehicle_Damage_bin
   - Vehicle_Age_Ordinal
   - Age

## Technologies Used
- Python 3.x
- Key libraries:
  - pandas (data manipulation)
  - scikit-learn (modeling)
  - imbalanced-learn (SMOTE)
  - seaborn/matplotlib (visualization)

## Future Improvements
- Experiment with other classification algorithms
- Feature engineering with interaction terms
- Collect additional relevant features
- Implement cross-validation for more robust evaluation
- Deploy model as a web service