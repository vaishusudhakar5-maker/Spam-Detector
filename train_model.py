import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_csv("spam.csv")

# Create model
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train model
model.fit(df["text"], df["label"])

# Save model
joblib.dump(model, "spam_model.pkl")
print("Model saved as spam_model.pkl")
