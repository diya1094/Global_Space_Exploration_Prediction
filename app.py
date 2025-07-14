import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("Global_Space_Exploration_Dataset.csv")

# Load trained models
success_model = joblib.load("success_model.pkl")
duration_model = joblib.load("duration_model.pkl")

# Streamlit App Title
st.title("🚀 Space Mission Success & Duration Predictor")
st.markdown("Enter details about a mission to get predictions.")

# Input fields
country = st.selectbox("Country", sorted(data["Country"].unique()))
year = st.number_input("Year", min_value=1950, max_value=2030, value=2025)
mission_type = st.selectbox("Mission Type", sorted(data["Mission Type"].unique()))
launch_site = st.text_input("Launch Site", "Example City")
satellite_type = st.selectbox("Satellite Type", sorted(data["Satellite Type"].unique()))
budget = st.number_input("Budget (in Billion $)", min_value=0.0, value=10.0)
tech_used = st.selectbox("Technology Used", sorted(data["Technology Used"].unique()))
impact = st.selectbox("Environmental Impact", sorted(data["Environmental Impact"].unique()))
collab = st.text_input("Collaborating Countries", "USA, India")

# Make dataframe
user_input = pd.DataFrame([{
    "Country": country,
    "Year": year,
    "Mission Type": mission_type,
    "Launch Site": launch_site,
    "Satellite Type": satellite_type,
    "Budget (in Billion $)": budget,
    "Technology Used": tech_used,
    "Environmental Impact": impact,
    "Collaborating Countries": collab
}])

# Predict
if st.button("Predict 🚀"):
    pred_success = success_model.predict(user_input)[0]
    pred_duration = duration_model.predict(user_input)[0]
    
    st.subheader("🔮 Predictions:")
    st.write(f"**Success Rate (%)**: {pred_success:.2f}")
    st.write(f"**Duration (in Days)**: {pred_duration:.0f}")

# Visualize the dataset
st.write("---")
st.write("## 📊 Data Visualizations")

# Split data for prediction vs actual graph
X = data.drop(['Success Rate (%)', 'Duration (in Days)'], axis=1)
y = data['Success Rate (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = success_model.predict(X_test)

# 1. Success Rate Over the Years 
st.write("### 1. Success Rate Over the Years")
fig1 = plt.figure(figsize=(8, 4))
plt.plot(data['Year'], data['Success Rate (%)'], marker='o', linestyle='-', alpha=0.7, color='green')
plt.xlabel('Year')
plt.ylabel('Success Rate (%)')
plt.title('Mission Success Rate Over the Years')
plt.grid(True)
st.pyplot(fig1)

#  2. Feature Importance 
st.write("### 2. Feature Importance (Model-Based)")
if hasattr(success_model, 'feature_importances_'):
    importances = success_model.feature_importances_
    features = X.columns if hasattr(X, 'columns') else range(len(importances))

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, importances, color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Which Features Matter Most?")
    st.pyplot(fig2)
else:
    st.warning("The current model doesn't support feature importances (only tree-based models like Random Forest do).")

#  3. Actual vs Predicted 
st.write("### 3. Actual vs Predicted: Success Rate")
fig3 = plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')  # Diagonal line
plt.xlabel("Actual Success Rate")
plt.ylabel("Predicted Success Rate")
plt.title("Model Performance: Actual vs Predicted")
plt.grid(True)
st.pyplot(fig3)
