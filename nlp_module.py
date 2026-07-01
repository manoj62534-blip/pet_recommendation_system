"""
nlp_module.py
─────────────
NLP personality extractor using TF-IDF + cosine similarity.
Import extract_personality() from this module into app.py.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and build TF-IDF matrix once at import time
_df = pd.read_csv("advanced_pet_nlp_dataset_5000.csv")
_df["combined_text"] = _df["description"].astype(str)

_vectorizer   = TfidfVectorizer(stop_words="english")
_tfidf_matrix = _vectorizer.fit_transform(_df["combined_text"])


def extract_personality(user_text: str) -> dict:
    """
    Match free-text user input to the closest dataset row via cosine
    similarity and return personality + pet recommendation fields.
    """
    user_vec   = _vectorizer.transform([user_text.lower()])
    similarity = cosine_similarity(user_vec, _tfidf_matrix)
    best_idx   = int(similarity.argmax())
    row        = _df.iloc[best_idx]

    return {
        "personality":        str(row.get("personality_type", "Unknown")),
        "introversion_score": int(row["introversion_score"]),
        "emotional_need":     int(row["emotional_need"]),
        "overthinking":       int(row["overthinking"]),
        "activity_level":     int(row["activity_level"]),
        "social_need":        int(row["social_need"]),
        "companionship_need": int(row["companionship_need"]),
        "recommended_pet":    str(row["recommended_pet"]),
        "recommended_breed":  str(row["recommended_breed"]),
    }
