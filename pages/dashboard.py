import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Superstore Dashboard", layout="wide")

# Check if user is logged in
if not st.session_state.get('logged_in', False):
    st.warning("Please log in to access the dashboard")
    st.stop()

# Title
st.title("Superstore Sales and Profit Prediction Dashboard")

# Sidebar with logout button
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    # Add logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = None
        st.rerun()

# Function to preprocess uploaded CSV for prediction
def preprocess_data(df, model_type="sales"):
    """
    Preprocess the uploaded CSV to match the features used in the trained model.
    model_type: 'sales' or 'profit' to handle differences in feature engineering.
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()

    # Strip column names to remove any whitespace
    df_processed.columns = df_processed.columns.str.strip()

    # Handle missing values by filling with mode for categorical and mean for numerical
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    # Convert 'Order Date' to datetime
    df_processed['Order Date'] = pd.to_datetime(df_processed['Order Date'], errors='coerce')

    if model_type == "sales":
        # Sales model: One-hot encoding and date features (Order_Year, Order_Month, Order_Day)
        df_processed['Order_Year'] = df_processed['Order Date'].dt.year
        df_processed['Order_Month'] = df_processed['Order Date'].dt.month
        df_processed['Order_Day'] = df_processed['Order Date'].dt.day
        if 'Ship Date' in df_processed.columns:
            df_processed['Ship Date'] = pd.to_datetime(df_processed['Ship Date'], errors='coerce')
            df_processed['Ship_Year'] = df_processed['Ship Date'].dt.year
            df_processed['Ship_Month'] = df_processed['Ship Date'].dt.month
            df_processed['Ship_Day'] = df_processed['Ship Date'].dt.day

        # One-hot encode categorical columns
        categorical_cols = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode', 'State']
        categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

        # Drop unnecessary columns
        columns_to_drop = [
            'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name',
            'City', 'Postal Code', 'Country', 'Order Date', 'Ship Date', 'Profit', 'Sales'
        ]
        df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], inplace=True)

        # Ensure all expected features are present
        expected_features = joblib.load(f'models/{model_type}_model.pkl').feature_names_in_
        missing_cols = [col for col in expected_features if col not in df_processed.columns]
        for col in missing_cols:
            df_processed[col] = 0
        df_processed = df_processed[expected_features]

    else:
        # Profit model: LabelEncoder and date features (Year, Month, Day)
        df_processed['Year'] = df_processed['Order Date'].dt.year
        df_processed['Month'] = df_processed['Order Date'].dt.month
        df_processed['Day'] = df_processed['Order Date'].dt.day

        # Load label encoders
        le_dict = joblib.load('models/profit_label_encoders.pkl')
        cat_cols = df_processed.select_dtypes(include='object').columns

        # Apply LabelEncoder to categorical columns
        for col in cat_cols:
            if col in le_dict:
                # Handle unseen categories by mapping to mode
                df_processed[col] = df_processed[col].apply(lambda x: x if x in le_dict[col].classes_ else df_processed[col].mode()[0])
                df_processed[col] = le_dict[col].transform(df_processed[col])
            else:
                df_processed.drop(columns=[col], inplace=True)

        # Drop Order Date
        df_processed.drop(columns=['Order Date'], inplace=True)

        # Ensure all expected features are present
        expected_features = joblib.load('models/profit_model.pkl').feature_names_in_
        missing_cols = [col for col in expected_features if col not in df_processed.columns]
        for col in missing_cols:
            df_processed[col] = 0
        df_processed = df_processed[expected_features]

    return df_processed

# Load models and encoders
try:
    sales_model = joblib.load('models/sales_model.pkl')
    profit_model = joblib.load('models/profit_model.pkl')
    profit_le_dict = joblib.load('models/profit_label_encoders.pkl')
except FileNotFoundError:
    st.error("Model or encoder files not found. Ensure 'sales_model.pkl', 'profit_model.pkl', and 'profit_label_encoders.pkl' are in the 'models/' folder.")
    st.stop()

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Save original data for comparison
        display_columns = ['Order Date', 'Product Name', 'Category', 'Sub-Category', 'Region', 'State', 'City', 'Sales', 'Quantity', 'Discount', 'Profit']
        original_display = df[display_columns].copy().reset_index()
        
        # Display raw data
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Tabs for different predictions
        tab1, tab2 = st.tabs(["Sales Prediction", "Profit Prediction"])

        with tab1:
            st.subheader("Sales Prediction")
            try:
                # Preprocess for sales model
                df_sales_processed = preprocess_data(df, model_type="sales")
                
                # Make predictions
                sales_predictions = sales_model.predict(df_sales_processed)
                
                # Create comparison table
                comparison_df = original_display.copy()
                comparison_df['Predicted Sales'] = np.round(sales_predictions, 2)
                comparison_df['Error'] = comparison_df['Sales'] - comparison_df['Predicted Sales']
                
                # Display comparison table
                st.write("Sales Predictions with Original Data")
                st.dataframe(comparison_df[['Order Date', 'Product Name', 'Category', 'Sub-Category', 'Region', 'State', 'City', 'Sales', 'Quantity', 'Discount', 'Profit', 'Predicted Sales', 'Error']].head(20))

                # Plot predictions
                st.write("Distribution of Predicted Sales")
                fig, ax = plt.subplots()
                sns.histplot(sales_predictions, bins=30, kde=True, ax=ax)
                ax.set_xlabel('Predicted Sales')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Predicted Sales')
                st.pyplot(fig)

                # Scatter plot of actual vs predicted sales
                st.write("Actual vs Predicted Sales")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=comparison_df['Sales'], y=comparison_df['Predicted Sales'], ax=ax)
                ax.set_xlabel('Actual Sales')
                ax.set_ylabel('Predicted Sales')
                ax.set_title('Actual vs Predicted Sales')
                ax.grid(True)
                st.pyplot(fig)

                # Download button for comparison table
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download Sales Predictions",
                    data=csv,
                    file_name="sales_predictions_labeled.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error in sales prediction: {str(e)}")

        with tab2:
            st.subheader("Profit Prediction")
            try:
                # Preprocess for profit model
                df_profit_processed = preprocess_data(df, model_type="profit")
                
                # Make predictions
                profit_predictions = profit_model.predict(df_profit_processed)
                
                # Create comparison table
                comparison_df = original_display.copy()
                comparison_df['Predicted Profit'] = np.round(profit_predictions, 2)
                comparison_df['Error'] = comparison_df['Profit'] - comparison_df['Predicted Profit']
                
                # Display comparison table
                st.write("Profit Predictions with Original Data")
                st.dataframe(comparison_df[['Order Date', 'Product Name', 'Category', 'Sub-Category', 'Region', 'State', 'City', 'Sales', 'Quantity', 'Discount', 'Profit', 'Predicted Profit', 'Error']].head(20))

                # Plot predictions
                st.write("Distribution of Predicted Profits")
                fig, ax = plt.subplots()
                sns.histplot(profit_predictions, bins=30, kde=True, ax=ax)
                ax.set_xlabel('Predicted Profit')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Predicted Profits')
                st.pyplot(fig)

                # Scatter plot of actual vs predicted profits
                st.write("Actual vs Predicted Profits")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=comparison_df['Profit'], y=comparison_df['Predicted Profit'], ax=ax)
                ax.set_xlabel('Actual Profit')
                ax.set_ylabel('Predicted Profit')
                ax.set_title('Actual vs Predicted Profit')
                ax.grid(True)
                st.pyplot(fig)

                # Download button for comparison table
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download Profit Predictions",
                    data=csv,
                    file_name="profit_predictions_labeled.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error in profit prediction: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to see predictions.")