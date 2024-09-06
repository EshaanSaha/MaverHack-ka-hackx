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
    analysis_results, high_regions_summary = analyze_defects({k: v for k, v in input_values.items() if k != 'BlinkState'})

    # Display the analysis results
    st.header("Analysis Results")
    for region, result in analysis_results.items():
        st.write(f"{region}: {result['alert']}")
        st.write(f"   {result['defects']}")

    # Display summary of high readings by region
    if high_regions_summary:
        st.header("High Reading Summary")
        for region_group, summary in high_regions_summary.items():
            st.write(summary)
    else:
        st.write("No high readings detected.")

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
