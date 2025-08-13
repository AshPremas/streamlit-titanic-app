import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
def load_data():
    df = pd.read_csv("Data/Titanic-Dataset.csv")
    df.columns = df.columns.str.strip()
    return df

def load_model():
    with open("Random_Forest_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

df = load_data()
model = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualisation", "Model Prediction", "Model Performance"])

# Home Page
if menu == "Home":
    st.title("Titanic Survival Prediction App")
    st.markdown("""
    ### Project Description
    This application predicts the survival chances of Titanic passengers based on their details. You can explore the dataset, visualise patterns, and get a real-time prediction.
    """)

# Data Exploration Section
elif menu == "Data Exploration":
    st.header('Data Exploration Section')
    st.markdown('### Dataset Overview')
    st.write(f"The dataset contains **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
    st.write(f"Columns and their data types:")
    st.write(df.dtypes)
    st.markdown('### Sample Data')
    st.write(df.head())
    st.markdown('### Interactive Data Filtering')
    selected_columns = st.multiselect('Select columns to display:', df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[selected_columns])

# Visualisation Section
elif menu == "Visualisation":
    st.header('Visualisation Section')
    st.markdown('### Survival by Gender')
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Sex', hue='Survived', ax=ax1)
    ax1.set_title('Survival Count by Gender')
    ax1.set_xlabel('Sex')
    ax1.set_ylabel('Number of Passengers')
    st.pyplot(fig1)

    st.markdown('### Survival by Passenger Class')
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=ax2)
    ax2.set_title('Survival Count by Pclass')
    ax2.set_xlabel('Passenger Class')
    ax2.set_ylabel('Number of Passengers')
    st.pyplot(fig2)

    st.markdown('### Age Distribution')
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax3)
    ax3.set_title('Age Distribution of Passengers')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)

# Model Prediction Section
elif menu == "Model Prediction":
    st.header('Model Prediction Section')

    def user_input_features():
        st.subheader('Enter Passenger Details')
        pclass = st.selectbox('Pclass', sorted(df['Pclass'].unique()))
        sex = st.selectbox('Sex', df['Sex'].unique())
        age = st.number_input('Age', 0, 100, 30)
        sibsp = st.slider('SibSp', 0, 8, 0)
        parch = st.slider('Parch', 0, 6, 0)
        fare = st.number_input('Fare', 0, 1000, 50)
        embarked = st.selectbox('Embarked', df['Embarked'].dropna().unique())
        data = {'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': embarked}
        features = pd.DataFrame(data, index=[0])
        return features

    df_input = user_input_features()

    def preprocess_input(df_input, model):
        df_processed = df_input.copy()
        
        # Handle categorical variables
        df_processed['Sex'] = df_processed['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        df_processed = pd.get_dummies(df_processed, columns=['Embarked'], prefix='Embarked', drop_first=True)

        # Align columns with the training data, excluding 'PassengerId'
        train_cols = [col for col in model.feature_names_in_ if col != 'PassengerId']
        df_processed = df_processed.reindex(columns=train_cols, fill_value=0)
        return df_processed

    processed_input = preprocess_input(df_input, model)
    
    st.subheader('User Input Features')
    st.write(processed_input)

    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            try:
                prediction = model.predict(processed_input)
                prediction_proba = model.predict_proba(processed_input)
                st.subheader('Prediction')
                survival_status = 'Survived' if prediction[0] == 1 else 'Did not survive'
                st.success(f'The passenger is predicted to: **{survival_status}**')
                st.subheader('Prediction Probability')
                st.write(f'Survival Probability: {prediction_proba[0][1]:.2f}')
                st.write(f'Non-survival Probability: {prediction_proba[0][0]:.2f}')
            except ValueError as e:
                st.error(f"Prediction Error: {e}. Please ensure all input fields are valid.")

# Model Performance Section
elif menu == "Model Performance":
    st.header('Model Performance Section')
    st.markdown("""
    ### Evaluation Metrics

    - **Accuracy:** 0.8380
    - **Precision:** 0 - 0.84 , 1 - 0.84
    - **Recall:** 0 - 0.90 , 1 - 0.76
    - **F1 Score:** 0 - 0.87 , 1 - 0.79
                
    ### Confusion Matrix
        [[94 11]
        [18 56]]

    ### Model Comparison
        Best Model: Random Forest with Accuracy: 0.8380
    """)