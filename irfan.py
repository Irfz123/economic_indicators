# # streamlit_app.py
# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.graph_objs as go
# import plotly.express as px
# from statsmodels.tsa.arima.model import ARIMA

# st.title("Economic Indicators & Inflation Dashboard")

# @st.cache_data
# def load_data():
#     # Load the dataset (ensure 'economic_indicators.csv' is in your working directory)
#     df = pd.read_csv('economic_indicators.csv')
#     # Convert 'Year' to datetime for proper time series handling
#     df['Date'] = pd.to_datetime(df['Year'], format='%Y')
#     df.sort_values('Date', inplace=True)
#     return df
    
# # Load data
# data = load_data()

# st.subheader("Dataset Overview")
# st.write("This dataset includes economic indicators for various countries over multiple years.")
# st.dataframe(data.head())

# # Country selection
# countries = data['Country'].unique().tolist()
# selected_country = st.selectbox("Select a Country", countries)

# # Filter data for the selected country
# country_data = data[data['Country'] == selected_country]

# if country_data.empty:
#     st.error("No data available for the selected country.")
# else:
#     st.subheader(f"Economic Trends for {selected_country}")
    
#     # Plotting historical trends for key indicators
#     indicators = ["GDP (in billion USD)", "Inflation Rate (%)", "Unemployment Rate (%)", "Economic Growth (%)"]
#     for indicator in indicators:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=country_data['Date'], y=country_data[indicator],
#             mode='lines+markers', name=indicator))
#         fig.update_layout(
#             title=f"{indicator} Over Time",
#             xaxis_title="Year", yaxis_title=indicator)
#         st.plotly_chart(fig)
    
#     # Correlation analysis on numeric columns (for the selected country)
#     st.subheader("Correlation Matrix of Economic Indicators")
#     numeric_cols = ["GDP (in billion USD)", "Inflation Rate (%)", "Unemployment Rate (%)", "Economic Growth (%)"]
#     corr_matrix = country_data[numeric_cols].corr()
#     fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
#     st.plotly_chart(fig_corr)
    
#     # Forecasting Inflation using ARIMA
#     st.subheader("Inflation Forecasting with ARIMA")
#     forecast_period = st.slider("Forecast Horizon (Years)", 1, 10, 5)
    
#     # Prepare the time series data for Inflation Rate
#     ts_data = country_data.set_index('Date')["Inflation Rate (%)"]
    
#     try:
#         # Fit an ARIMA model (order parameters may be tuned further)
#         model = ARIMA(ts_data, order=(1, 1, 1))
#         model_fit = model.fit()
#         forecast = model_fit.forecast(steps=forecast_period)
#     except Exception as e:
#         st.error(f"Error fitting ARIMA model: {e}")
#         forecast = None
    
#     if forecast is not None:
#         forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(years=1),
#                                        periods=forecast_period, freq='Y')
#         forecast_series = pd.Series(forecast, index=forecast_dates)
        
#         # Plot historical and forecasted inflation
#         st.subheader("Historical vs. Forecasted Inflation")
#         fig_forecast = go.Figure()
#         fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines+markers', name='Historical Inflation'))
#         fig_forecast.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines+markers', name='Forecasted Inflation'))
#         fig_forecast.update_layout(title="Inflation Forecast", xaxis_title="Year", yaxis_title="Inflation Rate (%)")
#         st.plotly_chart(fig_forecast)
        
#         st.subheader("Forecast Data")
#         forecast_df = forecast_series.reset_index().rename(columns={'index': 'Year', 0: 'Forecasted Inflation'})
#         forecast_df['Year'] = forecast_df['Year'].dt.year  # Display year only
#         st.dataframe(forecast_df)
        
#         # Actionable insights based on forecast vs. historical mean inflation
#         historical_mean = ts_data.mean()
#         forecast_mean = forecast_series.mean()
        
#         st.subheader("Actionable Insights")
#         if forecast_mean > historical_mean:
#             st.write(f"**Insight:** The forecasted average inflation rate ({forecast_mean:.2f}%) is higher than the historical average ({historical_mean:.2f}%).")
#             st.write("This suggests rising inflation, which could erode purchasing power and signal increased economic pressure.")
#             st.markdown("- **Policymakers:** Consider implementing measures to control inflation, such as adjusting interest rates or tightening fiscal policy.")
#             st.markdown("- **Investors:** Look into inflation-protected securities or diversify into assets that traditionally perform well during inflationary periods.")
#             st.markdown("- **Businesses:** Evaluate cost structures and supply chain strategies to mitigate the impact of rising prices.")
#         else:
#             st.write(f"**Insight:** The forecasted average inflation rate ({forecast_mean:.2f}%) is lower than the historical average ({historical_mean:.2f}%).")
#             st.write("This trend could indicate easing inflationary pressures, potentially leading to more stable economic conditions.")
#             st.markdown("- **Policymakers:** Consider stimulating economic growth through targeted investments or fiscal incentives if the trend continues.")
#             st.markdown("- **Investors:** Reassess portfolios to capitalize on potentially lower inflation and improved market conditions.")
#             st.markdown("- **Businesses:** Explore opportunities for expansion and innovation in a lower-inflation environment.")
#         st.write("These insights offer a foundation for making informed decisions tailored to the economic context of the selected country.")

'''


SECOND PROJECT


'''

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Sleep Health & Lifestyle Analysis and Prediction Dashboard")

@st.cache_data
def load_data():
    # Load the dataset (ensure 'sleep_health_lifestyle.csv' is in your working directory)
    df = pd.read_csv('sleep_health_lifestyle.csv')
    st.write("Columns in dataset:", df.columns.tolist())
    return df

# Load the data
data = load_data()

st.subheader("Dataset Overview")
st.dataframe(data.head())

# Data Preprocessing
st.subheader("Data Preprocessing")
# Drop rows with missing values and create a copy to avoid SettingWithCopyWarning
data_clean = data.dropna().copy()

# Encode Gender using .loc to avoid SettingWithCopyWarning
if 'Gender' in data_clean.columns:
    le_gender = LabelEncoder()
    data_clean.loc[:, 'Gender_enc'] = le_gender.fit_transform(data_clean['Gender'])
else:
    st.error("Gender column not found.")

# Encode BMI Category
if 'BMI Category' in data_clean.columns:
    le_bmi = LabelEncoder()
    data_clean.loc[:, 'BMI_Category_enc'] = le_bmi.fit_transform(data_clean['BMI Category'])
else:
    st.error("BMI Category column not found.")

# Show cleaned data preview
st.dataframe(data_clean.head())

# --- Exploratory Data Analysis (EDA) ---
st.subheader("Exploratory Data Analysis")

# Distribution of Quality of Sleep
if 'Quality of Sleep' in data_clean.columns:
    fig1 = px.histogram(data_clean, x='Quality of Sleep', color='Quality of Sleep', title="Distribution of Quality of Sleep")
    st.plotly_chart(fig1)
else:
    st.error("Quality of Sleep column not found.")

# Plot distributions of numeric features
numeric_features = ['Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Daily Steps', 'Age']
for col in numeric_features:
    if col in data_clean.columns:
        fig = px.histogram(data_clean, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig)
    else:
        st.warning(f"{col} not found in dataset.")

# Correlation Heatmap: numeric features and encoded categorical features
corr_features = numeric_features + ['Gender_enc', 'BMI_Category_enc']
corr = data_clean[corr_features].corr()
fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
st.plotly_chart(fig_corr)

# --- Predictive Modeling ---
st.subheader("Predictive Modeling: Predicting Quality of Sleep")
if 'Quality of Sleep' not in data_clean.columns:
    st.error("The dataset must contain a 'Quality of Sleep' column.")
else:
    # Encode Quality of Sleep if it's categorical
    if data_clean['Quality of Sleep'].dtype == object:
        le_sleep = LabelEncoder()
        data_clean.loc[:, 'Quality_of_Sleep_enc'] = le_sleep.fit_transform(data_clean['Quality of Sleep'])
    else:
        data_clean.loc[:, 'Quality_of_Sleep_enc'] = data_clean['Quality of Sleep']

    # Define features and target
    feature_cols = ['Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Daily Steps', 'Age', 'Gender_enc', 'BMI_Category_enc']
    X = data_clean[feature_cols]
    y = data_clean['Quality_of_Sleep_enc']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {acc:.2%}")

    # Display confusion matrix using Seaborn
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    # Feature Importance
    st.write("Feature Importance:")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    st.bar_chart(importance)

# --- Interactive Prediction Form ---
st.subheader("Predict Your Sleep Quality")
st.write("Enter your lifestyle details to predict your sleep quality.")

input_sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
input_physical_activity = st.number_input("Physical Activity Level (sessions per week)", min_value=0, max_value=14, value=3)
input_stress_level = st.number_input("Stress Level (scale 1-10)", min_value=1, max_value=10, value=5)
input_daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000, step=100)
input_age = st.number_input("Age", min_value=10, max_value=100, value=30)
input_gender = st.selectbox("Gender", options=["Male", "Female"])

# Use unique BMI Category options from the dataset for the selectbox
bmi_options = sorted(data_clean['BMI Category'].unique().tolist())
input_bmi_category = st.selectbox("BMI Category", options=bmi_options)

# Now encode the selected BMI category
input_bmi_cat_enc = int(le_bmi.transform([input_bmi_category])[0])

# Encode inputs consistent with training encoding
input_gender_enc = 0 if input_gender == "Female" else 1

if st.button("Predict Sleep Quality"):
    new_input = pd.DataFrame({
        'Sleep Duration': [input_sleep_duration],
        'Physical Activity Level': [input_physical_activity],
        'Stress Level': [input_stress_level],
        'Daily Steps': [input_daily_steps],
        'Age': [input_age],
        'Gender_enc': [input_gender_enc],
        'BMI_Category_enc': [input_bmi_cat_enc]
    })
    prediction = model.predict(new_input)[0]
    predicted_label = le_sleep.inverse_transform([prediction])[0] if 'le_sleep' in locals() else prediction
    st.write(f"**Predicted Quality of Sleep:** {predicted_label}")

# --- Actionable Insights ---
st.subheader("Actionable Insights")
st.write("""
Based on the model and exploratory analysis, here are some insights:
- **Sleep Duration:** Adequate sleep is associated with better sleep quality.
- **Physical Activity & Daily Steps:** Higher physical activity and more daily steps are linked to improved sleep.
- **Stress Level:** Elevated stress levels can negatively impact sleep quality.
- **BMI & Age:** Maintaining a healthy BMI and managing age-related changes contribute to better sleep.
  
**Recommendations:**
- **For Individuals:** Optimize sleep hygiene by balancing sleep duration, physical activity, and stress management.
- **For Healthcare Providers:** Advise patients on lifestyle modifications that can improve sleep quality.
""")