import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Hotel Price Analysis", layout="wide")

# Input Data
data = {
    "Name of the Hotel": [
        "Rema Resort", "Nirvana", "Hotel Gakyidiana", "Dzi Pema", "Hotel Khamsum",
        "Hotel Dorjiling", "Nirvana Lodge", "Hotel Kanchi Grand", "Taktsang Paradise",
        "Naksel Boutique Hotel & Spa", "Hotel Lhayul", "Yang Bhutan Hotel", "Spirit of Bhutan Lodge",
        "Le Méridien , Riverfront", "Shomo Chuki Resort", "The Tiger Nest Camp"
    ],
    "Dzongkhag": [
        "Paro", "Paro", "Nearby Tachog Lhakhang", "Paro", "Paro", "Paro", "Paro",
        "Nearby Tachog Lhakhang", "Nearby Paro Taktsang", "Nearby Kyichu Lhakhang", "Nearby Kyichu Lhakhang",
        "Paro", "Paro", "Paro", "Paro", "Nearby Paro Taktsang"
    ],
    "Rate (Nu)": [
        4780, 3292, 3191, 3849, 2633, 5064, 2449, 15192, 5191, 30387, 3191, 3191,
        4221, 58233, 5241, 9909
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Add Inflation Adjustment Factor (2024 Inflation Rate: 1.35%)
inflation_rate = 1.0135
df['Adjusted Rate (Nu)'] = df['Rate (Nu)'] * inflation_rate

# Encoding Dzongkhag for machine learning
df['Dzongkhag_Encoded'] = pd.factorize(df['Dzongkhag'])[0]

# Prepare Data for Machine Learning
X = df[['Dzongkhag_Encoded']]  # Features
y = df['Adjusted Rate (Nu)']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Prices
df['Predicted Price (Nu)'] = model.predict(X)

# Streamlit App
st.title("Hotel Price Analysis")

# Sidebar
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show Raw Data", value=True)
show_adjusted = st.sidebar.checkbox("Show Inflation Adjusted Rates", value=True)
show_prediction = st.sidebar.checkbox("Show Predicted Prices", value=True)

# Display Raw Data
if show_data:
    st.subheader("Raw Data")
    st.dataframe(df[['Name of the Hotel', 'Dzongkhag', 'Rate (Nu)']])

# Display Inflation-Adjusted Rates
if show_adjusted:
    st.subheader("Inflation-Adjusted Rates")
    st.dataframe(df[['Name of the Hotel', 'Dzongkhag', 'Rate (Nu)', 'Adjusted Rate (Nu)']])

# Display Predicted Prices
if show_prediction:
    st.subheader("Predicted Prices")
    st.dataframe(df[['Name of the Hotel', 'Dzongkhag', 'Rate (Nu)', 'Adjusted Rate (Nu)', 'Predicted Price (Nu)']])
pip install streamlit pandas scikit-learn
streamlit run streamlit_app.py
