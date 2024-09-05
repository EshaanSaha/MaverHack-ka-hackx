import streamlit as st
import numpy as np
import pickle

# Load the pre-trained machine learning model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Calculate averages from EEG readings
def calculate_averages(readings):
    electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']
    averages = {}
    for i, electrode in enumerate(electrodes):
        average = np.mean([reading[i] for reading in readings])
        averages[electrode] = average
    return averages

# Evaluate defects based on EEG readings
def evaluate_eeg_defects(readings):
    averages = calculate_averages(readings)
    risk_assessment = {}
    
    for i, electrode in enumerate(['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']):
        electrode_readings = [reading[i] for reading in readings]
        average = averages[electrode]
        
        for reading in electrode_readings:
            if reading > average * 1.1:
                risk_assessment[(electrode, reading)] = 'High Reading - Potential Risk'
            elif reading < average * 0.9:
                risk_assessment[(electrode, reading)] = 'Low Reading - Potential Risk'
            else:
                risk_assessment[(electrode, reading)] = 'Normal Reading'

    return risk_assessment

# Input Tab UI using Streamlit
def main():
    # Add a title
    st.title("EEG Reading Analysis and ML Prediction")

    # Input fields for EEG readings
    st.header("Enter EEG Readings:")
    eeg_readings = []
    electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']

    # Collect readings for 3 samples
    for i in range(3):
        st.subheader(f"Sample {i + 1}")
        sample_readings = [st.number_input(f"Reading for {electrode}:", min_value=0.0, max_value=5000.0, value=0.0) for electrode in electrodes]
        eeg_readings.append(sample_readings)

    # Display the inputs
    st.write("Your EEG Readings:", eeg_readings)

    # Evaluate the defects
    if st.button("Evaluate Defects"):
        risk_assessment = evaluate_eeg_defects(eeg_readings)
        st.write("Risk Assessment:")
        for (electrode, reading), risk in risk_assessment.items():
            st.write(f"{electrode} ({reading} µV): {risk}")

    # Load the model
    model = load_model()

    # Input fields for ML prediction
    st.header("Enter Features for ML Prediction:")
    feature_1 = st.number_input("Enter the average value of frontal region:", min_value=0.0, max_value=5000.0, value=10.0)
    feature_2 = st.number_input("Enter the average value of occipital region:", min_value=0.0, max_value=5000.0, value=20.0)
    feature_3 = st.number_input("Enter the average value of parietal region:", min_value=0.0, max_value=5000.0, value=30.0)
    feature_4 = st.number_input("Enter the average value of front central region:", min_value=0.0, max_value=5000.0, value=40.0)
    feature_5 = st.number_input("Enter the average value of temporal region:", min_value=0.0, max_value=5000.0, value=50.0)
    
    input_features = [feature_1, feature_2, feature_3, feature_4, feature_5]
    
    # Display the inputs
    st.write("Your Input Features:", input_features)

    # Predict using the model
    if st.button("Predict"):
        prediction = model.predict(np.array(input_features).reshape(1, -1))
        st.success(f"The prediction result is: {prediction[0]}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
