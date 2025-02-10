import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load Data from Uploaded Excel File
def load_data(file):
    return pd.read_excel(file)


def train_model(data):
    X = data.iloc[:, :-1]  # Features (all columns except the last one)
    y = data.iloc[:, -1]   # Target (last column)

    # Ensure all non-numeric columns are encoded properly
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns  # Save feature names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = RandomForestClassifier(random_state=42)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    
    model.fit(X_train, y_train)
    
    # extra-----------
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance_df)
    
    # # -------------

    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)

    # return model, accuracy, feature_names
    
    # Predictions--------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Return the model, accuracies, and feature names
    return model, train_accuracy, test_accuracy, feature_names
    # ------------------

# Prediction Function
def predict_performance(model, input_data):
    input_data = pd.get_dummies(input_data, drop_first=True)  # Convert input data to numeric
    return model.predict(input_data)


# Streamlit App
def main():
    
    st.set_page_config(page_title="Employee Performance Prediction")


    st.title("Employee Performance Prediction")

    uploaded_file = st.file_uploader("C:/Users/Parashuram Singh/OneDrive/Desktop/Employee data FINAL.xlsx", type=["xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data Preview:", data.head(60))

        # Train Model
        # st.write("### Training Model")
        # -------------------------
        # # model, accuracy = train_model(data)
        # model, accuracy, feature_names = train_model(data)  # Get feature names from training

        
        # st.success(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")
        # --------------------
        # Train Model
        st.write("### Training Model")
        model, train_accuracy, test_accuracy, feature_names = train_model(data)

        st.success(f"Model trained successfully!")
        st.write(f"**Training Accuracy:** {train_accuracy * 100:.2f}%")
        st.write(f"**Testing Accuracy:** {test_accuracy * 100:.2f}%")

        # Input Form for Prediction
        st.write("### Predict Employee Performance")
        input_data = {}
        for col in data.columns[:-1]:
            # unique_values = data[col].dropna().unique()  # Get unique values for the column
            unique_values = sorted(data[col].dropna().unique())
            input_data[col] = st.selectbox(f"Select {col}", options=unique_values)


        # Predict Performance Button Logic===
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None  # Initialize session state for prediction    
        # =====
        if st.button("Predict Performance"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df, drop_first=True)

            # Align input_df with training feature names
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0  # Add missing columns with default value 0
            input_df = input_df[feature_names]  # Reorder columns to match training

            # -----------------
            print("Input Data for Prediction:", input_df)
            print("Prediction Probabilities:", model.predict_proba(input_df))
            print("Predicted Class:", model.predict(input_df))
            # -----------------
            
            prediction = predict_performance(model, input_df)
            st.session_state.prediction_result = f"Predicted Performance: {prediction[0]}"  # Store result in session state
            
        # Display the prediction result
        if st.session_state.prediction_result:
            st.success(st.session_state.prediction_result)    
            # st.success(f"Predicted Performance: {prediction[0]}")

        # ===================
        if "save_status" not in st.session_state:
            st.session_state.save_status = None  # Initialize session state for save status
        # ==================
        if st.button("Save Inputs to New Excel"):
            input_df = pd.DataFrame([input_data])  # Create a DataFrame for the current input

            # Perform one-hot encoding for the input data
            input_df = pd.get_dummies(input_df, drop_first=True)

            # Align input features with the training feature names
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0  # Add missing columns with default value 0
            input_df = input_df[feature_names]  # Reorder columns to match training

            # Add the prediction value as a new column
            prediction = predict_performance(model, input_df)  # Get the prediction
            input_data["Predicted Performance"] = prediction[0]  # Add prediction to the original input data

            # File name for storing data
            new_file = "C:/Users/Parashuram Singh/OneDrive/Desktop/Book1.xlsx"

            try:
                # Check if the file exists
                existing_data = pd.read_excel(new_file)
                combined_data = pd.concat([existing_data, pd.DataFrame([input_data])], ignore_index=True)  # Append new data
            except FileNotFoundError:
                # If file doesn't exist, create it with the current input
                combined_data = pd.DataFrame([input_data])

            # Save the combined data to the file
            combined_data.to_excel(new_file, index=False)
            # ===============
            st.session_state.save_status = f"Inputs and prediction saved to {new_file}"  # Store save status

        # Display the save status
        if st.session_state.save_status:
            st.success(st.session_state.save_status)

        # Compare Performance
        st.write("### Compare Employee Performance")
        compare_id = st.text_input("Enter Employee Number to Compare")

        if st.button("Compare Performance"):
            new_data = pd.read_excel("C:/Users/Parashuram Singh/OneDrive/Desktop/Book1.xlsx")
            old_performance = data[data[data.columns[0]] == compare_id]
            new_performance = new_data[new_data[new_data.columns[0]] == compare_id]

            if not old_performance.empty and not new_performance.empty:
                st.write("Old Performance:", old_performance)
                st.write("New Performance:", new_performance)
            else:
                st.error("Employee Number not found in one or both files.")
        
    
if __name__ == "__main__":
    main()




#stremlit run "app_path"
#cd appdata  cd roaming    cd python   cd python312    cd scripts .\streamlit run "app_path"  