# 🧠 Mental Tiredness Prediction System

## 📌 Project Overview

The **Mental Tiredness Prediction System** is a Machine Learning application that predicts a person's mental tiredness score based on work habits, sleep quality, workload, lifestyle, and environmental factors.

The project includes data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment using **Streamlit** to provide an interactive web application.

---

## 🎯 Problem Statement

Mental fatigue affects productivity, decision-making, and overall well-being. This project aims to predict the level of mental tiredness using machine learning so that users can understand their mental health status and take preventive measures.

---

## 🚀 Features

- Data preprocessing
- Outlier treatment
- Categorical encoding
- Feature scaling
- Feature engineering
- Feature selection
- Hyperparameter tuning using GridSearchCV
- Ridge Regression model
- Interactive Streamlit web application
- Mental tiredness prediction
- Personalized recommendations
- Animated UI effects
- Download prediction as CSV

---

## 📂 Project Structure

```
Mental_Tiredness_Project/
│
├── app/
│   └── app.py
│
├── data/
│   └── mental_tiredness_score_prediction_dataset.csv.csv
│
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset Features

The model uses the following input features:

- Mood
- Work Environment
- Work Type
- Number of Decisions Made
- Context Switch Count
- Notifications Received
- Screen Time
- Deep Work Time
- Task Complexity
- Caffeine Intake
- Break Frequency
- Sleep Hours
- Deep Sleep Percentage
- Hydration
- Noise Level
- Temperature
- Workload Score

Target Variable:

- Mental Tiredness Score

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Streamlit Extras
- Git
- GitHub

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Outlier Treatment using IQR Method
- Ordinal Encoding
- One-Hot Encoding
- Standard Scaling
- Feature Engineering
- Feature Selection

---

## 🤖 Machine Learning Models Tested

The following regression models were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression (Selected Model)

---

## 🔍 Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** on the Ridge Regression model.

Parameter tuned:

- Alpha

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| R² Score | 0.767 |
| MAE | 4.69 |
| RMSE | 5.90 |

The Ridge Regression model was selected because it provided the best balance between training and testing performance with minimal overfitting.

---

## 🖥️ Streamlit Application

The Streamlit application allows users to:

- Enter personal and work-related information
- Predict Mental Tiredness Score
- View recommendations
- Experience interactive UI animations
- Download prediction results

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Mental-Tiredness-Prediction.git
```

Move into the project directory

```bash
cd Mental-Tiredness-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python src/train.py
```

This generates:

- model.pkl
- preprocessor.pkl
- scaler.pkl

inside the **models** folder.

---

## ▶️ Run the Application

```bash
streamlit run app/app.py
```

---

## 📋 Prediction Categories

| Score | Category |
|--------|----------|
| 0 – 20 | Very Low Mental Tiredness |
| 21 – 40 | Low Mental Tiredness |
| 41 – 60 | Moderate Mental Tiredness |
| 61 – 80 | High Mental Tiredness |
| 81 – 100 | Very High Mental Tiredness |

---

## 📌 Future Improvements

- Deploy on Streamlit Community Cloud
- Docker Support
- Model Monitoring
- User Authentication
- Database Integration
- Real-time Prediction API
- Deep Learning Models
- Explainable AI (SHAP)

---

## 👨‍💻 Author

**Dandoti Sagar**

B.Tech – Electronics and Communication Engineering

Machine Learning Project

---

## 📜 License

This project is created for educational and learning purposes.
