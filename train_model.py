"""
train_model.py
──────────────
Run this ONCE to train the Random Forest models and save all .pkl files.
Usage:  python train_model.py

Output files:
  pet_model.pkl      – predicts recommended pet type
  breed_model.pkl    – predicts recommended breed
  encoders.pkl       – LabelEncoders for categorical input columns
  pet_encoder.pkl    – LabelEncoder for pet labels
  breed_encoder.pkl  – LabelEncoder for breed labels
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("pet_recommendation_dataset_5000.csv")

# ── Encode categorical input columns ─────────────────────────────────────────
# CSV columns: Age, House_Type, Budget, Free_Time_Hours,
#              Personality, Activity_Level, Noise_Tolerance, Experience
encoders = {}
CATEGORICAL_COLS = [
    "House_Type", "Budget", "Personality",
    "Activity_Level", "Noise_Tolerance", "Experience",
]

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── Features ──────────────────────────────────────────────────────────────────
X = df.drop(["Recommended_Pet", "Recommended_Breed"], axis=1)

# ── Pet type model ────────────────────────────────────────────────────────────
pet_encoder = LabelEncoder()
y_pet = pet_encoder.fit_transform(df["Recommended_Pet"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pet, test_size=0.2, random_state=42
)
pet_model = RandomForestClassifier(n_estimators=200, random_state=42)
pet_model.fit(X_train, y_train)

acc = pet_model.score(X_test, y_test)
print(f"Pet model accuracy : {acc:.2%}")

# ── Breed model ───────────────────────────────────────────────────────────────
breed_encoder = LabelEncoder()
y_breed = breed_encoder.fit_transform(df["Recommended_Breed"])

breed_model = RandomForestClassifier(n_estimators=200, random_state=42)
breed_model.fit(X, y_breed)

# ── Save everything ───────────────────────────────────────────────────────────
pickle.dump(pet_model,     open("pet_model.pkl",     "wb"))
pickle.dump(breed_model,   open("breed_model.pkl",   "wb"))
pickle.dump(encoders,      open("encoders.pkl",      "wb"))
pickle.dump(pet_encoder,   open("pet_encoder.pkl",   "wb"))
pickle.dump(breed_encoder, open("breed_encoder.pkl", "wb"))

print("All models saved successfully!")
print("Features used:", list(X.columns))
