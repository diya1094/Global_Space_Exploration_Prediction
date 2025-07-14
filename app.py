import streamlit as st
import pandas as pd
import joblib
# Load trained models
success_model = joblib.load("success_model.pkl")
duration_model = joblib.load("duration_model.pkl")

st.title("🚀 Space Mission Success & Duration Predictor")
st.markdown("Enter details about a mission to get predictions.")

# Input fields
country = st.selectbox("Country", ["India", "USA", "Russia", "China", "Japan", "Israel", "UAE", "Other"])
year = st.number_input("Year", min_value=1950, max_value=2030, value=2025)
mission_type = st.selectbox("Mission Type", ["Manned", "Unmanned"])
launch_site = st.text_input("Launch Site", "Example City")
satellite_type = st.selectbox("Satellite Type", ["Communication", "Weather", "Spy", "Other"])
budget = st.number_input("Budget (in Billion $)", min_value=0.0, value=10.0)
tech_used = st.selectbox("Technology Used", ["Nuclear Propulsion", "Solar Propulsion", "AI Navigation", "Traditional Rocket", "Other"])
impact = st.selectbox("Environmental Impact", ["Low", "Medium", "High"])
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