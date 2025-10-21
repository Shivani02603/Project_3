#!/usr/bin/env python3
"""
fix_shivani_model.py
Uses YOUR actual student_habits_performance.csv to train the model
Follows Shivani's AI_study.ipynb notebook EXACTLY
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

print("="*70)
print(" "*15 + "CREATING MODEL FROM YOUR CSV DATA")
print(" "*10 + "(Following Shivani's AI_study.ipynb Exactly)")
print("="*70)

# ============================================================================
# STEP 1: Import Libraries & Load Dataset (from notebook)
# ============================================================================
print("\nStep 1: Loading your CSV file...")

try:
    df = pd.read_csv("student_habits_performance.csv", encoding='utf-8')
    print(f"  ✓ Loaded {len(df)} student records")
    print(f"  ✓ Columns: {len(df.columns)}")
    print(f"\n  First few column names:")
    for col in df.columns[:5]:
        print(f"    - {col}")
except FileNotFoundError:
    print("  ✗ ERROR: student_habits_performance.csv not found!")
    print("  Please make sure the CSV file is in the same folder.")
    exit(1)
except Exception as e:
    print(f"  ✗ ERROR loading CSV: {e}")
    exit(1)

# ============================================================================
# STEP 2: Basic Data Cleaning (from notebook)
# ============================================================================
print("\nStep 2: Data cleaning...")

# Check for missing values
print(f"  Missing values before handling:")
missing = df.isnull().sum()
if missing.sum() > 0:
    for col in missing[missing > 0].index:
        print(f"    {col}: {missing[col]}")
else:
    print(f"    None!")

# Fill missing values in parental_education_level with most common value (mode)
if 'parental_education_level' in df.columns:
    if df['parental_education_level'].isnull().sum() > 0:
        df['parental_education_level'].fillna(df['parental_education_level'].mode()[0], inplace=True)
        print(f"  ✓ Filled missing parental_education_level values")

print(f"  ✓ Data cleaning complete")

# ============================================================================
# STEP 3: Feature Engineering (from notebook)
# ============================================================================
print("\nStep 3: Feature engineering...")

# Encode diet quality to numeric points
df['diet_points'] = df['diet_quality'].map({'Good': 3, 'Fair': 2, 'Poor': 1})

# Total distractions = social media + Netflix
df['total_distractions'] = df['social_media_hours'] + df['netflix_hours']

# Focus index = study hours / (total distractions + small constant)
df['focus_index'] = df['study_hours_per_day'] / (df['total_distractions'] + 1e-6)

# Health score = sleep + exercise + diet
df['health_score'] = df['sleep_hours'] + df['exercise_frequency'] + df['diet_points']

print(f"  ✓ Created: total_distractions")
print(f"  ✓ Created: focus_index")
print(f"  ✓ Created: health_score")

# ============================================================================
# STEP 4: Create Target Column (from notebook)
# ============================================================================
print("\nStep 4: Creating target labels...")

# Define function for labeling procrastination risk (EXACT from notebook)
def label_risk(score):
    if score < 50:
        return 'HIGH'
    elif score < 70:
        return 'MEDIUM'
    else:
        return 'LOW'

# Apply function
df['procrastination_risk'] = df['exam_score'].apply(label_risk)

# Encode risk as numeric for ML
df['risk_encoded'] = df['procrastination_risk'].map({'HIGH': 2, 'MEDIUM': 1, 'LOW': 0})

low_count = sum(df['risk_encoded']==0)
med_count = sum(df['risk_encoded']==1)
high_count = sum(df['risk_encoded']==2)

print(f"  Risk distribution in your data:")
print(f"    LOW:    {low_count:4d} ({low_count/len(df)*100:5.1f}%)")
print(f"    MEDIUM: {med_count:4d} ({med_count/len(df)*100:5.1f}%)")
print(f"    HIGH:   {high_count:4d} ({high_count/len(df)*100:5.1f}%)")

# ============================================================================
# STEP 5: Encode Text Columns (from notebook)
# ============================================================================
print("\nStep 5: Encoding categorical features...")

# Define encoding maps (EXACT from notebook)
encode_map = {
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'part_time_job': {'Yes': 1, 'No': 0},
    'extracurricular_participation': {'Yes': 1, 'No': 0},
    'diet_quality': {'Poor': 1, 'Fair': 2, 'Good': 3},
    'internet_quality': {'Poor': 1, 'Average': 2, 'Good': 3},
    'parental_education_level': {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3}
}

# Apply encodings
df.replace(encode_map, inplace=True)

print(f"  ✓ Encoded 6 categorical columns")

# ============================================================================
# STEP 6: Choose Final Features (from notebook)
# ============================================================================
print("\nStep 6: Selecting features...")

# EXACT feature list from Shivani's notebook
selected_features = [
    'study_hours_per_day',
    'social_media_hours',
    'netflix_hours',
    'total_distractions',
    'focus_index',
    'sleep_hours',
    'mental_health_rating',
    'attendance_percentage',
    'diet_quality',
    'exercise_frequency',
    'part_time_job',
    'age'
]

X = df[selected_features]
y = df['risk_encoded']

print(f"  Feature matrix shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Features: {len(selected_features)}")

# ============================================================================
# STEP 7: Train-Test Split (from notebook)
# ============================================================================
print("\nStep 7: Train-test split...")

# EXACT parameters from notebook
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples:  {X_test.shape[0]}")

# ============================================================================
# STEP 8: Evaluate Model (Train + Test) (from notebook)
# ============================================================================
print("\nStep 8: Training Random Forest model...")

# EXACT model parameters from Shivani's notebook
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)
print("  ✓ Model training complete!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n  🎯 Model Accuracy: {acc:.2%}")

# Classification report
print("\n  📊 Classification Report:")
report = classification_report(y_test, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH'])
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print("           Predicted")
print("           LOW  MED  HIGH")
print(f"  Actual LOW  {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
print(f"         MED  {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
print(f"         HIGH {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")

# ============================================================================
# STEP 9: Save Model & Results (from notebook)
# ============================================================================
print("\nStep 9: Saving model...")

# EXACT saving method from Shivani's notebook
joblib.dump(model, "trained_model.pkl")
print("  💾 Model saved as trained_model.pkl")

# Save results
with open("model_training_results.txt", "w") as f:
    f.write(f"Model Training Results\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Testing samples: {X_test.shape[0]}\n")
    f.write(f"Features used: {len(selected_features)}\n\n")
    f.write(f"Classification Report:\n")
    f.write(report)

print("  ✓ Results saved to model_training_results.txt")

# ============================================================================
# STEP 10: Try Sample Predictions (from notebook)
# ============================================================================
print("\nStep 10: Testing with sample prediction...")

# EXACT sample from notebook
sample = pd.DataFrame([{
    'study_hours_per_day': 2,
    'social_media_hours': 4,
    'netflix_hours': 3,
    'total_distractions': 7,
    'focus_index': 2 / (7 + 1),
    'sleep_hours': 5,
    'mental_health_rating': 4,
    'attendance_percentage': 70,
    'diet_quality': 2,
    'exercise_frequency': 2,
    'part_time_job': 0,
    'age': 21
}])

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
risk_label = {0:'LOW', 1:'MEDIUM', 2:'HIGH'}[prediction]

print(f"\n  Sample Student:")
print(f"    Study: 2h/day, Distractions: 7h, Attendance: 70%")
print(f"    📘 Predicted Risk: {risk_label}")
print(f"    Probabilities:")
print(f"      LOW:    {probabilities[0]*100:5.1f}%")
print(f"      MEDIUM: {probabilities[1]*100:5.1f}%")
print(f"      HIGH:   {probabilities[2]*100:5.1f}%")

# ============================================================================
# VERIFICATION: Test the saved model
# ============================================================================
print("\n" + "="*70)
print("VERIFICATION: Testing saved model...")
print("="*70)

try:
    loaded_model = joblib.load("trained_model.pkl")
    print(f"✓ Model loaded successfully")
    print(f"  Type: {type(loaded_model).__name__}")
    print(f"  Has predict(): {hasattr(loaded_model, 'predict')}")
    print(f"  Has predict_proba(): {hasattr(loaded_model, 'predict_proba')}")
    
    # Test with loaded model
    test_pred = loaded_model.predict(sample)[0]
    test_proba = loaded_model.predict_proba(sample)[0]
    test_risk = {0:'LOW', 1:'MEDIUM', 2:'HIGH'}[test_pred]
    
    print(f"\n  Testing loaded model:")
    print(f"    Prediction: {test_risk}")
    print(f"    Matches original: {test_pred == prediction}")
    print(f"    Probabilities match: {np.allclose(probabilities, test_proba)}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")

print("\n" + "="*70)
print(" "*20 + "MODEL CREATION COMPLETED!")
print("="*70)

print(f"""
✅ SUCCESS! Model trained on YOUR actual data!

📊 Your Data Stats:
   • Total students: {len(df)}
   • Training set: {len(X_train)} students
   • Test set: {len(X_test)} students
   • Model accuracy: {acc:.1%}

📁 Files Created:
   1. trained_model.pkl - Ready for Tushar's backend
   2. model_training_results.txt - Detailed metrics

🎯 Next Steps:
   → Run: python test_prediction.py
   → Run: python complete_system_demo.py

🎉 Your backend is now ready with REAL data!
""")