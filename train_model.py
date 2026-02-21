import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# Load
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(columns=['id'])

# Fix BMI
df['bmi'] = df['bmi'].replace('N/A', np.nan)
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Drop rare gender
df = df[df['gender'] != 'Other'].copy()

# Encode categoricals
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Upsample minority class (stroke=1) to balance dataset
df_majority = df[df.stroke == 0]
df_minority = df[df.stroke == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop(columns=['stroke'])
y = df_balanced['stroke']
feature_cols = list(X.columns)

# Scale & Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(classification_report(y_test, model.predict(X_test)))

# Save artifacts
with open("svm_stroke_model.pkl", "wb") as f: pickle.dump(model, f)
with open("stroke_scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("stroke_encoders.pkl", "wb") as f: pickle.dump(encoders, f)
with open("stroke_feature_cols.pkl", "wb") as f: pickle.dump(feature_cols, f)

print("✅ All artifacts saved!")
