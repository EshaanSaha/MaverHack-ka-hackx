import streamlit as st
import numpy as np
import pickle

# Load the pre-trained machine learning model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Input Tab UI using Streamlit
def main():
    # Add a title
    st.title("Numeric Input for ML Model")

    # Create input fields for numeric data
    st.header("Enter Numeric Features:")
    feature_1 = st.number_input("Enter the average value of frontal region: ", min_value=0.0, max_value=5000.0, value=10.0)
    feature_2 = st.number_input("Enter the average value of occipital region:", min_value=0.0, max_value=5000.0, value=20.0)
    feature_3 = st.number_input("Enter the average value of parietal region:", min_value=0.0, max_value=5000.0, value=30.0)
    feature_4 = st.number_input("Enter the average value of front central region:", min_value=0.0, max_value=5000.0, value=40.0)
    feature_5 = st.number_input("Enter the average value of temporal region:", min_value=0.0, max_value=5000.0, value=50.0)
    
    # Store inputs in a list
    input_features = [feature_1, feature_2, feature_3, feature_4, feature_5]
    
    # Display the inputs
    st.write("Your Input Features:", input_features)

    # Load the model
    model = load_model()

    # Predict using the model
    if st.button("Predict"):
        prediction = model.predict(np.array(input_features).reshape(1, -1))
        st.success(f"The prediction result is: {prediction[0]}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
