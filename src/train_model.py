# best_train.py
# ✅ Balanced training: accept all survived cases and match equal non-survived cases

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# ====== Paths ======
DATA_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/data/dataset_med.csv"
MODEL_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/model.pkl"
SCALER_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/scaler.pkl"
FEATURES_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/feature_names.pkl"
REPORT_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/outputs/classification_report.txt"

# ====== Load & Prepare Data ======
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
df.drop(['id', 'diagnosis_date', 'end_treatment_date', 'country'], axis=1, inplace=True)

# ====== Balance Dataset: take all survived (1s) and equal non-survived (0s) ======
df_1 = df[df['survived'] == 1]  # All survived
count_1 = len(df_1)
df_0 = df[df['survived'] == 0].sample(count_1, random_state=42)  # Equal non-survived

df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# ====== Categorical Columns ======
cat_cols = ['gender', 'cancer_stage', 'family_history', 'smoking_status',
            'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']
df_encoded = pd.get_dummies(df_balanced, columns=cat_cols, drop_first=True)

# ====== Features and Target ======
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

# ====== Save Feature Names ======
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
joblib.dump(X.columns.tolist(), FEATURES_PATH)

# ====== Scale Features ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)

# ====== Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ====== Train Model ======
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ====== Evaluate Model ======
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# ====== Save Artifacts ======
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

with open(REPORT_PATH, "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

print("✅ Balanced model trained and saved with evaluation report.")


'''
# best_train.py
# ✅ Final version: Best training configuration using full dataset with Random Forest

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# ====== Paths ======
DATA_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/data/dataset_med.csv"
MODEL_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/model.pkl"
SCALER_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/scaler.pkl"
FEATURES_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/feature_names.pkl"
REPORT_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/outputs/classification_report.txt"

# ====== Load & Prepare Data ======
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
df.drop(['id', 'diagnosis_date', 'end_treatment_date', 'country'], axis=1, inplace=True)

# ====== Categorical Columns ======
cat_cols = ['gender', 'cancer_stage', 'family_history', 'smoking_status',
            'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ====== Features and Target ======
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

# ====== Save Feature Names ======
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
joblib.dump(X.columns.tolist(), FEATURES_PATH)

# ====== Scale Features ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)

# ====== Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ====== Train Model ======
model = RandomForestClassifier(
    n_estimators=300,                # More trees
    max_depth=None,                  # Let it grow fully
    class_weight='balanced_subsample',  # Handle imbalance
    random_state=42,
    n_jobs=-1                        # Use all CPU cores
)
model.fit(X_train, y_train)

# ====== Evaluate Model ======
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# ====== Save Artifacts ======
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

with open(REPORT_PATH, "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

print("✅ Best model trained and saved with evaluation report.")'''