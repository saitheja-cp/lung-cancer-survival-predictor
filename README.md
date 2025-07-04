# 🩺 Lung Cancer Survival Prediction

## 🎯 Objective
Build a machine learning system that predicts the survival of lung cancer patients based on their medical, demographic, and treatment-related data.

## 📊 Dataset
The dataset consists of 890,000 patient records and includes features like:
- Age, BMI, cholesterol level
- Cancer stage, smoking history, comorbidities
- Type of treatment and its duration
- Survival outcome (0 = Did not survive, 1 = Survived)

## 🔧 Model Pipeline
- Preprocessed using pandas and one-hot encoding
- Trained a Random Forest classifier with scikit-learn
- Used StandardScaler for feature scaling
- Model and scaler are saved using joblib for deployment

## 🖥 Streamlit Web App
A simple GUI is included to enter new patient data and receive a survival prediction.

### 🏃 To Run the App:
```bash
pip install -r requirements.txt
streamlit run app/predict_app.py
```

## 📁 Folder Structure
```
lung_cancer_survival_project/
├── data/               # Raw dataset
├── saved_models/       # Saved model & scaler
├── app/                # Streamlit GUI
├── outputs/            # Evaluation metrics or plots
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

## 📦 Requirements
```bash
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
openpyxl
streamlit
```

## 📬 Author
- **Name**: C.P. Sai Theja  
- **Email**: [cpsaitheja@gmail.com](mailto:cpsaitheja@gmail.com)

> This project showcases full ML lifecycle integration — from preprocessing and model training to GUI-based prediction.
