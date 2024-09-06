import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load average values (replace with actual file or method to obtain these)
def load_average_values():
    # Example of average values; replace with your actual average values
    return {
        'AF3': 4200, 'AF4': 4200, 'F7': 4200, 'F3': 4200, 'FC5': 4200,
        'FC6': 4200, 'F4': 4200, 'T7': 4200, 'T8': 4200, 'P7': 4200,
        'P8': 4200, 'O1': 4200, 'O2': 4200
    }

# Analyze defects based on EEG values
def analyze_defects(values):
    analysis = {}
    threshold_percentage = 0.10  # 10% deviation threshold
    max_value = 5500
    expected_value = 4200
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

# Load and count user focus data from CSV
def load_user_counts():
    df = pd.read_csv('user_counts.csv')
    df['eyeDetection'] = df['eyeDetection'].astype(int)  # Ensure correct data type

    focused_count = df[df['eyeDetection'] == 1].shape[0]
    not_focused_count = df[df['eyeDetection'] == 0].shape[0]

    return focused_count, not_focused_count

# Plot comparison of user data and average values
def plot_comparison(user_values, avg_values):
    regions = list(user_values.keys())
    user_vals = [user_values[region] for region in regions]
    avg_vals = [avg_values.get(region, 0) for region in regions]

    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(regions))

    bar1 = ax.bar(index, user_vals, bar_width, label='User Data', color='b', alpha=0.6)
    bar2 = ax.bar(index + bar_width, avg_vals, bar_width, label='Average Values', color='r', alpha=0.6)

    ax.set_xlabel('EEG Regions')
    ax.set_ylabel('Values')
    ax.set_title('EEG Values Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(regions, rotation=90)
    ax.legend()

    plt.tight_layout()
    return fig

# Plot user focus counts pie chart
def plot_focus_counts_pie_chart(focused_count, not_focused_count):
    labels = ['Focused', 'Not Focused']
    sizes = [focused_count, not_focused_count]
    colors = ['#ff9999','#66b3ff']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.set_title('User Focus Distribution')
    return fig

# Main function to run the Streamlit app
def main():
    st.title("EEG Input Analysis and ML Model Prediction")

    # Display image
    st.image('scan1.jpg', use_column_width=True)

    # Input fields for numeric data
    st.header("Enter the EEG values:")
    regions = ['AF3', 'AF4', 'F7', 'F3', 'FC5', 'FC6', 'F4', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']
    input_values = {region: st.number_input(f"{region} (Region)", min_value=0, max_value=5500, value=4450, step=1) for region in regions}

    # Input field for blink state
    blink_state = st.selectbox("Select Blink State", options=[0, 1])

    # Store blink state in input values
    input_values['BlinkState'] = blink_state

    # Analyze defects based on the input values
    analysis_results = analyze_defects({k: v for k, v in input_values.items() if k != 'BlinkState'})

    # Display the analysis results
    st.header("Analysis Results")
    for region, result in analysis_results.items():
        st.write(f"{region}: {result['alert']}")
        st.write(f"   {result['defects']}")

    # Load the model and average values
    model = load_model()
    avg_values = load_average_values()

    # Predict using the model
    if st.button("Predict"):
        # Prepare input data for prediction (including blink state)
        input_data = np.array(list(input_values.values())).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Display focus classification based on model prediction
        attention_span = "Focused" if prediction == 1 else "Not Focused"
        st.write(f"Focus Classification: {attention_span}")

        # Plot comparison of user data and average model values
        fig = plot_comparison(input_values, avg_values)
        st.pyplot(fig)

        # Load and display user counts
        focused_count, not_focused_count = load_user_counts()
        fig2 = plot_focus_counts_pie_chart(focused_count, not_focused_count)
        st.pyplot(fig2)

# Run the Streamlit app
if __name__ == '__main__':
    main()
