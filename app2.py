import streamlit as st
import numpy as np
import pickle


# Load the pre-trained machine learning model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Function to analyze potential defects based on input values
def analyze_defects(values):
    analysis = {}
    threshold_percentage = 0.08  # 8% deviation threshold
    max_value = 5500
    expected_value = max_value / 2  # Assuming the middle of the range as expected value
    deviation = expected_value * threshold_percentage
    lower_bound = expected_value - deviation
    upper_bound = expected_value + deviation

    for region, value in values.items():
        if value > upper_bound:
            analysis[region] = {
                'alert': "High Alert",
                'defects': "Potential defect: Abnormal activity that might indicate anxiety, hyperactivity, stress, or other cognitive issues."
            }
        elif value < lower_bound:
            analysis[region] = {
                'alert': "Low Alert",
                'defects': "Potential defect: Abnormal activity that could be related to depression, attention issues, cognitive decline, or other neurological problems."
            }
        else:
            analysis[region] = {
                'alert': "Normal",
                'defects': "Reading is within the expected range."
            }

    return analysis


# Streamlit application
def main():
    st.title("EEG Input Analysis and ML Model Prediction")

    # Input fields for numeric data
    st.header("Enter the EEG values:")

    # Input fields for each brain region with max value of 5500
    AF3 = st.number_input("AF3 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    AF4 = st.number_input("AF4 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    F7 = st.number_input("F7 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    F3 = st.number_input("F3 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    FC5 = st.number_input("FC5 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    FC6 = st.number_input("FC6 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    F4 = st.number_input("F4 (Frontal Region)", min_value=0, max_value=5500, value=2750, step=1)
    T7 = st.number_input("T7 (Temporal Region)", min_value=0, max_value=5500, value=2750, step=1)
    T8 = st.number_input("T8 (Temporal Region)", min_value=0, max_value=5500, value=2750, step=1)
    P7 = st.number_input("P7 (Parietal Region)", min_value=0, max_value=5500, value=2750, step=1)
    P8 = st.number_input("P8 (Parietal Region)", min_value=0, max_value=5500, value=2750, step=1)
    O1 = st.number_input("O1 (Occipital Region)", min_value=0, max_value=5500, value=2750, step=1)
    O2 = st.number_input("O2 (Occipital Region)", min_value=0, max_value=5500, value=2750, step=1)

    # Store inputs in a dictionary
    input_values = {
        'AF3': AF3,
        'AF4': AF4,
        'F7': F7,
        'F3': F3,
        'FC5': FC5,
        'FC6': FC6,
        'F4': F4,
        'T7': T7,
        'T8': T8,
        'P7': P7,
        'P8': P8,
        'O1': O1,
        'O2': O2
    }

    # Analyze defects based on the input values
    analysis_results = analyze_defects(input_values)

    # Display the analysis results
    st.header("Analysis Results")
    for region, result in analysis_results.items():
        st.write(f"{region}: {result['alert']}")
        st.write(f"   {result['defects']}")

    # Load the model
    model = load_model()

    # Predict using the model
    if st.button("Predict"):
        # Prepare input data for prediction (assuming the model expects a specific format)
        input_data = np.array(list(input_values.values())).reshape(1, -1)
        prediction = model.predict(input_data)
        st.success(f"The prediction result is: {prediction[0]}")


# Run the Streamlit app
if __name__ == '__main__':
    main()
