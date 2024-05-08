import streamlit as st
import pandas as pd
import io
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# Function to render the sidebar menu
def render_sidebar():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Go to", ("Home","Data Cleaning","Chart Visualization","Model Building"))
    return selected_page
@st.cache_data
# Function to render the home page
def render_home():
    st.title("Codsoft Data Science InternShip ")
    st.header("Wecome to Task 01")
    st.subheader("Here's a bouquet &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

    st.subheader('''Use the **Titanic dataset** to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data.''')
    st.subheader('''The dataset typically used for this project contains information about individual passengers, such as their *age*, *gender*, *ticket class*, *fare*, *cabin*, and whether or not they survived.''')

# Function to render the chart visualization page
def render_chart_visualization():
        AVAILABLE_CHARTS = ['Scatter Plot', 'Line Chart', 'Area Chart', 'Bar Chart', 'Violin Plot', 'Box Plot', 'Histogram', 'Pie Chart', 'Heatmap']
        st.title("Chart Visualization")
        st.write("Here you can visualize charts.")
    
        # Upload CSV file
        st.sidebar.subheader("Upload Dataset")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
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
            chart_type = st.selectbox('Select Chart Type', AVAILABLE_CHARTS)
        
            # Generate and display chart based on selected type
            if chart_type == 'Heatmap':
                fig = generate_heatmap(data[selected_features])
            else:
                fig = generate_chart(data, selected_features, chart_type)
            st.plotly_chart(fig)
        else:
            st.write('Please upload a CSV file.')

# Function to generate chart using Plotly Express based on chart type
def generate_chart(data, selected_features, chart_type):
    if chart_type == 'Scatter Plot':
        fig = px.scatter(data, x=data.index, y=selected_features, title='Scatter Plot')
    elif chart_type == 'Line Chart':
        fig = px.line(data, x=data.index, y=selected_features, title='Line Chart')
    elif chart_type == 'Area Chart':
        fig = px.area(data, x=data.index, y=selected_features, title='Area Chart')
    elif chart_type == 'Bar Chart':
        fig = px.bar(data, x=data.index, y=selected_features, title='Bar Chart')
    elif chart_type == 'Violin Plot':
        fig = px.violin(data, y=selected_features, box=True, title='Violin Plot')
    elif chart_type == 'Box Plot':
        fig = px.box(data, y=selected_features, title='Box Plot')
    elif chart_type == 'Histogram':
        fig = px.histogram(data, x=selected_features, title='Histogram')
    elif chart_type == 'Pie Chart':
        fig = px.pie(data, values=selected_features[0], names=data.index, title='Pie Chart')
    else:
        fig = px.scatter(data, x=data.index, y=selected_features, title='Scatter Plot')  # Default to scatter plot
    fig.update_layout(width=900, height=800)
    return fig

# Function to generate heatmap using Plotly Express
def generate_heatmap(data):
    fig = px.imshow(data.corr(), title='Heatmap Correlation Matrix',color_continuous_scale='YlGnBu')
    fig.update_layout(width=900, height=800)  # Change color scale to cover a wide range
    return fig

# Function to generate a downloadable CSV from a DataFrame
def get_csv_download_link(dataframe, filename):
    # Create a CSV file in memory
    csv_buffer = io.StringIO()  # in-memory text stream
    dataframe.to_csv(csv_buffer, index=False)  # Save DataFrame to CSV
    csv_buffer.seek(0)  # Reset the buffer position to the start
    
    # Create a download button
    st.download_button(
        label="Download Cleaned Data",
        data=csv_buffer.getvalue(),  # Get CSV content
        file_name=filename,  # Desired file name for download
        mime="text/csv",  # Content type
    )
    
# Example function to convert CSR matrix to DataFrame
def csr_to_dataframe(sparse_matrix, column_names):
    # Convert sparse matrix to dense array
    dense_array = sparse_matrix.toarray()
    
    # Create a DataFrame from the dense array
    dataframe = pd.DataFrame(dense_array, columns=column_names)
    
    return dataframe

def render_Data_Cleaning():
    st.title("Data Cleaning")
    st.write("Here you can Cleaning the Data Set of Missing Value and Null column.")
    # Upload CSV file
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    # If file uploaded
    if uploaded_file is not None:
    # Read CSV file into DataFrame, reading only the first 100 records
        data = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.write("Uploaded DataFrame :")
        st.write(data)
        st.subheader("Information about the Dataset")
        st.write(data.info())
        st.subheader("Summary Statistics for numerical Columns")
        st.write(data.describe())
        st.write(data.isnull().sum())
        st.subheader("Select feature to be drop")
        selected_features = st.multiselect("Select Features", data.columns)
        
        if not selected_features:
            st.error("Please select at least one feature.")
            return
        
        # Check if selected features are present in the dataset
        if not all(feature in data.columns for feature in selected_features):
            st.error("One or more selected features are not present in the dataset.")
            return

        if selected_features:
            # Drop selected features
            data.drop(columns=selected_features, inplace=True)
            st.success(f"Dropped {len(selected_features)} columns.")

        # Split data into features (X) and target (y)
        X = data.drop(columns=["Survived"])
        y = data["Survived"]

        # Define numerical and categorical features
        numerical_features = ["Age", "Fare"]
        categorical_features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

        # Define preprocessing steps for numerical and categorical data
        numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine preprocessing steps for numerical and categorical data
        preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
        ])

        # Apply preprocessing to features
        X_preprocessed = preprocessor.fit_transform(X)

        # Display preprocessed data (can be displayed partially)
        st.subheader("Preprocessed Data")
        st.write("Shape:", X_preprocessed.shape)
        st.write("Sample of Preprocessed Data:")
        # Display preprocessed features
        st.write(X_preprocessed)

        st.subheader("Download Cleaned Data")
        get_csv_download_link(X_preprocessed, "cleaned_titanic_data.csv")  # Call the download function
    else:
        st.warning("Please upload a dataset to proceed.")



def select_features(data, selected_features):
    return data[selected_features]

def check_survived_column(data):
    if "Survived" not in data.columns:
        st.error("Error: The dataset does not contain the 'Survived' column. Please ensure you are using the correct dataset.")
        return False
    return True


def render_Model_Building():
    st.title("Titanic Survival Model Building")
    # Upload the dataset
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    st.write("Upload Data Set for Model Building")
    # If dataset is uploaded
    if uploaded_file is not None:
        # Load the data
        data = pd.read_csv(uploaded_file)
        if not check_survived_column(data):
            # If the 'Survived' column is not found, we can return early to prevent further errors
            return
        st.subheader("Titanic Dataset")
        st.write(data)
        st.subheader("Titanic Describe")
        st.write(data.describe())
        st.subheader("Titanic Feature")
        st.write(data.columns)
        

        if data.empty:
            st.error("Dataset is empty.")
            return
        
        # Feature selection
        st.subheader("Feature Selection")
        selected_features = st.multiselect("Select Features", data.columns)
        
        if not selected_features:
            st.error("Please select at least one feature.")
            return
        
        # Check if selected features are present in the dataset
        if not all(feature in data.columns for feature in selected_features):
            st.error("One or more selected features are not present in the dataset.")
            return
        
        scaler = MinMaxScaler()

        # Apply normalization to numerical features
        data[selected_features] = scaler.fit_transform(data[selected_features])

        # Display the normalized dataset
        st.write(data.head())
        # Check if 'Survived' column exists before dropping it
        if 'Survived' in data.columns:
            # Perform feature selection
            selected_data = select_features(data, selected_features)

            # Split data into features (X) and target (y)
            y = selected_data["Survived"]
            X = selected_data.drop(columns=["Survived"])
            

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.subheader("Model Selection")
            selected_model = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree"])

            # Initialize and train the selected model
            if selected_model == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            elif selected_model == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif selected_model == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            elif selected_model == "Support Vector Machine":
                model = SVC(random_state=42)
            elif selected_model == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)

            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)

            # Display the accuracy
            st.subheader("Model Evaluation")
            st.write("Accuracy:", accuracy)
        else:
            st.error("Dataset does not contain 'Survived' column.")


# Main function to render the selected page
def main():
    selected_page = render_sidebar()

    if selected_page == "Home":
        render_home()
    elif selected_page == "Chart Visualization":
        render_chart_visualization()
    elif selected_page == "Model Building":
        render_Model_Building()  
    elif selected_page == "Data Cleaning":
        render_Data_Cleaning()      

# Entry point of the app
if __name__ == "__main__":
    main()
