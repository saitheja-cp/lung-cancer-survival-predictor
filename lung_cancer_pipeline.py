# lung_cancer_pipeline_from_scratch.py
# ✅ End-to-end pipeline with correlation check

import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ====== Paths ======
DATA_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/data/dataset_med.csv"
MODEL_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/saved_models/model.pkl"
SCALER_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/saved_models/scaler.pkl"
FEATURES_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/saved_models/feature_names.pkl"
PROB_CSV_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/outputs/full_data_with_probabilities.csv"
CATEGORY_ANALYSIS_XLSX = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/outputs/category_analysis.xlsx"
REPORT_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/outputs/classification_report.txt"
CORR_PATH = "........lung-cancer-survival-predictor-main/lung-cancer-survival-predictor-main/outputs/correlation_heatmap.png"

# ====== Load & preprocess dataset ======
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
df.drop(['id', 'diagnosis_date', 'end_treatment_date', 'country'], axis=1, inplace=True)

# ====== Correlation Analysis ======
cat_cols = ['gender', 'cancer_stage', 'family_history', 'smoking_status',
            'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']
df_encoded_all = pd.get_dummies(df, columns=cat_cols, drop_first=True)

corr = df_encoded_all.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
os.makedirs(os.path.dirname(CORR_PATH), exist_ok=True)
plt.savefig(CORR_PATH)
plt.close()
print(f"✅ Correlation heatmap saved → {CORR_PATH}")

# ====== Balance dataset ======
df_1 = df[df['survived'] == 1]
df_0 = df[df['survived'] == 0].sample(len(df_1), random_state=42)
df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# ====== Encode Balanced Data ======
df_encoded = pd.get_dummies(df_balanced, columns=cat_cols, drop_first=True)
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
joblib.dump(X.columns.tolist(), FEATURES_PATH)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)

# ====== Evaluation ======
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
with open(REPORT_PATH, "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

# ====== Full Dataset Prediction ======
df_full = pd.read_csv(DATA_PATH)
df_full.columns = df_full.columns.str.strip()
df_full['diagnosis_date'] = pd.to_datetime(df_full['diagnosis_date'])
df_full['end_treatment_date'] = pd.to_datetime(df_full['end_treatment_date'])
df_full['treatment_duration'] = (df_full['end_treatment_date'] - df_full['diagnosis_date']).dt.days
df_base = df_full.drop(['id', 'diagnosis_date', 'end_treatment_date', 'country'], axis=1).copy()
df_encoded_full = pd.get_dummies(df_base.drop('survived', axis=1), columns=cat_cols, drop_first=True)
X_full = df_encoded_full.reindex(columns=X.columns, fill_value=0)
X_full_scaled = scaler.transform(X_full)

probs = model.predict_proba(X_full_scaled)[:, 1]
df_base['survival_probability'] = probs
df_base.to_csv(PROB_CSV_PATH, index=False)

# ====== Category-wise analysis ======
with pd.ExcelWriter(CATEGORY_ANALYSIS_XLSX) as writer:
    for col in ['smoking_status', 'cancer_stage', 'family_history', 'treatment_type', 'gender', 'hypertension']:
        group = df_full[[col]].copy()
        group['survival_probability'] = probs
        result = group.groupby(col)['survival_probability'].mean().reset_index()
        result.to_excel(writer, sheet_name=col, index=False)

print("✅ Full pipeline completed: Model trained, probabilities predicted, correlations checked, and analysis saved.")
