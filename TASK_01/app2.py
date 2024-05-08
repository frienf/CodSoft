import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Streamlit app
st.title('Plotly Chart with Multiselect')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# If file uploaded
if uploaded_file is not None:
    # Read CSV file into DataFrame, reading only the first 100 records
    data = pd.read_csv(uploaded_file, nrows=100)

    # Display uploaded data
    st.write("Uploaded DataFrame (First 100 records):")
    st.write(data)

    # Multiselect dropdown to select values
    selected_values = st.multiselect('Select Values', options=data.columns, default=[data.columns[0]])

    # Define colors for each selected feature
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'yellow']

    # Generate Plotly chart based on selected values
    if selected_values:
        fig = go.Figure()
        for i, value in enumerate(selected_values):
            fig.add_trace(go.Bar(x=data.index, y=data[value], name=value, marker_color=colors[i % len(colors)]))

        fig.update_layout(title='Bar Chart for Selected Features',
                          xaxis_title='Index',
                          yaxis_title='Value',
                          height=600, width=800)  # Adjust chart size

        st.plotly_chart(fig)
    else:
        st.write('Please select at least one value.')
else:
    st.write('Please upload a CSV file.')
