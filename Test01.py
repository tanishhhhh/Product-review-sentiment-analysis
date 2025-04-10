import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Set Streamlit page title
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")
st.title("📊 Product Reviews Sentiment Analysis")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Upload Dataset", "Data Processing", "Train Model", "Evaluate Model", "Predict Sentiment"])

# Upload dataset
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.sidebar.error("Please upload a dataset to proceed.")
    st.stop()

# Rename columns
st.subheader("🔄 Renaming Columns")
df.rename(columns={
    "Id": "Record ID",
    "ProductId": "Product ID",
    "UserId": "User ID",
    "ProfileName": "User Profile Name",
    "HelpfulnessNumerator": "Helpfulness Numerator",
    "HelpfulnessDenominator": "Helpfulness Denominator",
    "Score": "Product Rating",
    "Time": "Review Time",
    "Summary": "Review Summary",
    "Text": "Review Text"
}, inplace=True)
st.success("Columns renamed successfully!")

# Convert Review Time
st.subheader("📅 Converting Review Time")
df["Review Time"] = pd.to_datetime(df["Review Time"], unit='s')
st.write(df[["Review Time"]].head())

# Assign sentiment labels
st.subheader("😊 Assigning Sentiment Labels")
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Product Rating"].apply(get_sentiment)
st.write(df[["Product Rating", "Sentiment"]].head())

# Sentiment Distribution Visualization
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Sentiment", palette="viridis", ax=ax)
st.pyplot(fig)

# Review Length Distribution
st.subheader("📏 Review Length Distribution")
df["Review Length"] = df["Review Text"].apply(lambda x: len(x.split()))
fig, ax = plt.subplots()
sns.histplot(df["Review Length"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Filter required columns
df = df[["Review Text", "Sentiment"]].dropna()

# Data preparation for training
st.subheader("⚙️ Preparing Data for Training")
X = df["Review Text"]
y = df["Sentiment"].map({"Positive": 2, "Neutral": 1, "Negative": 0})

# Train-test split
st.subheader("📊 Splitting Data into Training & Testing Sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display dataset sizes
st.write(f"🟢 Training Set Size: {X_train.shape[0]} reviews")
st.write(f"🔵 Testing Set Size: {X_test.shape[0]} reviews")

# Show sample data from Training & Testing sets
st.subheader("📌 Sample Reviews from Training & Testing Sets")

# Display a few training examples
st.write("### 🔹 Training Set (First 3 Samples)")
train_samples = pd.DataFrame({"Review Text": X_train[:3].values, "Sentiment": y_train[:3].values})
st.write(train_samples)

# Display a few testing examples
st.write("### 🔹 Testing Set (First 3 Samples)")
test_samples = pd.DataFrame({"Review Text": X_test[:3].values, "Sentiment": y_test[:3].values})
st.write(test_samples)

# Vectorization
st.subheader("🔠 Applying TF-IDF Vectorization")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Display TF-IDF matrix shape
st.write(f"TF-IDF matrix shape (Train): {X_train_tfidf.shape}")
st.write(f"TF-IDF matrix shape (Test): {X_test_tfidf.shape}")

# TF-IDF Feature Importance
st.subheader("🔠 Top 20 TF-IDF Words")
feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(vectorizer.idf_)[::-1][:20]
top_words = feature_array[tfidf_sorting]
top_idf_values = vectorizer.idf_[tfidf_sorting]

# Display words and their scores in a dataframe
top_tfidf_df = pd.DataFrame({"Word": top_words, "IDF Score": top_idf_values})
st.write(top_tfidf_df)

# Visualization
fig, ax = plt.subplots()
sns.barplot(x=top_words, y=top_idf_values, ax=ax, palette="coolwarm")
ax.set_xticklabels(top_words, rotation=45)
ax.set_ylabel("Inverse Document Frequency (IDF)")
st.pyplot(fig)

# Train model
st.subheader("🤖 Training Logistic Regression Model")
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)
st.success("Model training completed!")

# Save model & vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
st.sidebar.success("Model & Vectorizer saved!")

# Model evaluation
st.subheader("📉 Model Evaluation")
y_pred = model.predict(X_test_tfidf)
st.metric(label="Model Accuracy", value=f"{accuracy_score(y_test, y_pred):.2f}")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Sentiment Prediction UI
st.subheader("🔍 Predict Sentiment of a Review")
user_input = st.text_area("Enter a product review:")
if st.button("Analyze", use_container_width=True):
    if user_input:
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        sentiment_map = {2: "Positive", 1: "Neutral", 0: "Negative"}
        st.success(f"Predicted Sentiment: {sentiment_map[prediction]}")
    else:
        st.warning("Please enter a review before analyzing.")
