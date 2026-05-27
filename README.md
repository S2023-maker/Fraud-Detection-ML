# 🔍 Fraud Detection System

A Machine Learning powered web application built using Streamlit to detect fraudulent credit card transactions in real time.

---

# 🚀 Project Overview

This project predicts whether a transaction is:

- ✅ Genuine
- 🚨 Fraudulent

using transaction-related details such as:
- Transaction Amount
- Payment Method
- Transaction Type
- Device Used
- Account Age
- Previous Fraudulent History
- Transaction Frequency

The model is trained using Machine Learning algorithms and deployed as an interactive Streamlit web application.

---

<img width="1850" height="805" alt="image" src="https://github.com/user-attachments/assets/1d6bd5c0-c5c1-47fb-bef0-18f90113fc9b" />


# 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-Learn (SMOTE)

---
# 📊 Machine Learning Workflow
1. Data Preprocessing
Handling missing values
Feature encoding
Feature scaling
Outlier handling
2. Handling Imbalanced Data

Used SMOTE (Synthetic Minority Oversampling Technique) to balance fraud and non-fraud classes.

3. Model Training

Models compared:

Logistic Regression
Decision Tree
Random Forest
XGBoost
4. Best Model

✅ XGBoost achieved the best performance and was selected for deployment.

# 🖥️ Application Preview
**Input Features**
Transaction Amount
Transaction Time
Device Used
Payment Method
Transaction Type
Account Age
Previous Fraudulent Transactions

**Output**
Fraud Prediction
Fraud Probability
Risk Level

# Deployment link:  
https://fraud-detection-ml-rdta7ro2tr6oj5y4aglvor.streamlit.app/

# 📂 Project Structure

```bash
Fraud_detection_WebApp/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── fraud_xgb_model.pkl
│   └── preprocessor.pkl


    └── ML_project_Fraud_Detection_Complete.ipynb
