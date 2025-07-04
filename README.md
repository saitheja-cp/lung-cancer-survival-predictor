# 🩺 Lung Cancer Survival Prediction

## 🎯 Objective
Build a machine learning system that predicts the survival of lung cancer patients based on their medical, demographic, and treatment-related data.

## 📊 Dataset
### 📂 Step 1: Download Dataset
Click below to download the dataset directly:

👉 [Download dataset_med.csv](https://drive.google.com/uc?export=download&id=17zbgw3Ef_4SUJSr69sBUD1IPDxC1_0IZ)

### 📂 Step 2: Move the File to the `data/` Folder

After downloading the file:

1. Create a folder named `data` in the project root (if it doesn't exist already).
2. Move the downloaded `dataset_med.csv` file into the `data` folder.

Your final folder structure should look like this

lung-cancer-survival-predictor/

  ├── data/

      └── dataset_med.csv
  
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
├── app/                 # Streamlit GUI
├── data/                # Raw dataset
├── outputs/             # Evaluation metrics or plots
├── saved_models/        # Saved model & scaler
├── lung_cancer_pipeline # Code
├── README.md            # Project overview
└── requirements.txt     # Python dependencies 
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
