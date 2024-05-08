import streamlit as st
import plotly.express as px
import pandas as pd

# Streamlit app
st.title('Chart Visualization')

# Function to generate bar chart using Plotly Express
def generate_bar_chart(data, selected_features):
    fig = px.bar(data, x=data.index, y=selected_features, title='Bar Chart')
    fig.update_layout(width=900, height=800)
    return fig

# Function to generate line chart using Plotly Express
def generate_line_chart(data, selected_features):
    fig = px.line(data, x=data.index, y=selected_features, title='Line Chart')
    fig.update_layout(width=900, height=800)
    return fig

# Function to generate scatter plot using Plotly Express
def generate_scatter_plot(data, selected_features):
    fig = px.scatter(data, x=data.index, y=selected_features, title='Scatter Plot')
    fig.update_layout(width=900, height=800)
    return fig

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# If file uploaded
if uploaded_file is not None:
    # Read CSV file into DataFrame, reading only the first 100 records
    data = pd.read_csv(uploaded_file, nrows=100)

    # Display uploaded data
    st.write("Uploaded DataFrame (First 100 records):")
    st.write(data)

    # Select features to visualize
    selected_features = st.multiselect('Select Features', options=data.columns, default=[data.columns[0]])

    # Select chart type
    chart_type = st.selectbox('Select Chart Type', ['Bar Chart', 'Line Chart', 'Scatter Plot'])

    # Generate and display chart based on selected type
    if chart_type == 'Bar Chart':
        fig = generate_bar_chart(data, selected_features)
    elif chart_type == 'Line Chart':
        fig = generate_line_chart(data, selected_features)
    elif chart_type == 'Scatter Plot':
        fig = generate_scatter_plot(data, selected_features)

    st.plotly_chart(fig)
else:
    st.write('Please upload a CSV file.')
