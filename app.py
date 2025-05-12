import streamlit as st
import numpy as np
import joblib
import os

# Fallback model loader
def safe_load_model(file_name):
    try:
        return joblib.load(file_name)
    except Exception as e:
        st.warning(f"⚠️ Failed to load model {file_name}: {e}")
        return None

# Load trained models
morning_model = safe_load_model("morning_rf_model.pkl")
evening_model = safe_load_model("evening_rf_model.pkl")


# App title
st.title("🔵 Net Sales Predictor")

# Time of day selection with defined hour ranges
st.subheader("Select Time of Day for Prediction")
st.write("Morning hours: 8–14  |  Evening hours: 15–20")

time_period = st.radio("Choose the time period that best matches your data:", ["Morning", "Evening"])


# Feature inputs (integers only, step of 10)
hour = st.slider("Hour", min_value=8, max_value=20, step=1, value=8)
avg_order_6w = st.number_input("Avg Order Past 6 Weeks", min_value=0, step=10, value=50, format="%d")
avg_sales_6w = st.number_input("Avg Sales Past 6 Weeks", min_value=0, step=10, value=1000, format="%d")
avg_sales_last_year = st.number_input("Avg Sales Same Weeks Prev Year (6w)", min_value=0, step=10, value=950, format="%d")

# Predict button
if st.button("Predict Net Sales"):
    input_data = np.array([[hour, avg_order_6w, avg_sales_6w, avg_sales_last_year]])
    model = morning_model if time_period == "Morning" else evening_model
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Net Sales: ${prediction:,.2f}")

# ---------------------------------------------
# Optional: Upload new Excel to retrain model
# ---------------------------------------------
st.subheader(" Upload New Forecasting Data (Optional)")

uploaded_file = st.file_uploader("Upload an Excel file to update forecasts", type=["xlsx"])

def retrain_model_from_excel(file):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    try:
        df = pd.read_excel(file)

        # Drop columns as done originally
        cols_to_drop = ['Month', 'Day', 'Year', 'WeekNumber', 'ISO_Week', 'Order Count', 'IsHolidaySeason']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        features = ['Hour', 'Avg_Order_Past6Weeks', 'Avg_Sales_Past6Weeks', 'Avg_Sales_SameWeeks_PrevYear_6w']
        target = 'Net Sales'

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        return rf

    except Exception as e:
        st.error(f"❌ Error retraining model: {e}")
        return None

if uploaded_file is not None:
    st.success("✅ File uploaded. Retraining model now...")
    new_model = retrain_model_from_excel(uploaded_file)

    if new_model:
        if time_period == "Morning":
            morning_model = new_model
        else:
            evening_model = new_model
        st.info("🔁 Model retrained and ready to use for predictions.")
