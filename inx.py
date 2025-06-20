#import the necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import openpyxl, xlrd


# Load the data
url = ("http://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls")
df = pd.read_excel(url)

# Add title to the app
st.title("Employee Performance Prediction Model")

# Display the image
st.image("Employee Performance Prediction App.jpg")

# Add header
st.header("About.", divider="rainbow")

# Add paragraph
st.write("""Welcome to the Employee Performance Prediction App! 
         This application leverages machine learning to forecast employee performance ratings based on various factors. 
         By analyzing extensive datasets from INX Future Inc., the app empowers Human Resorces professionals and organizational leaders to gain insights into employee performance trends. 
         Through the predictive modeling technique used, Random Forest Classification and hyperparameter optimization, the app delivers the most accurate predictions, 
         enabling proactive decision-making in talent management, resource allocation, and performance improvement initiatives. 
         Whether you're a manager seeking to optimize team dynamics or an HR specialist aiming to enhance employee engagement, this app equips you with actionable insights to drive organizational success. 
         Explore the data, evaluate model performance, and make informed decisions to elevate your workforce's performance to new heights.""")

# Display exploratory data analysis (EDA)
st.header("Exploratory Data Analysis (EDA).", divider="rainbow")

if st.button("Show Dataset Info"):
    st.write("Dataset Info:", df.info())

if st.button("Number of Rows"):
    st.write("Number of Rows:", df.shape[0])

if st.button("Column Names"):
    st.write("Column Names:", df.columns.tolist())

if st.button("Data Types"):
    st.write("Data Types:", df.dtypes)

if st.button("Missing Values"):
    st.write("Missing Values:", df.isnull().sum())

if st.button("Statistical Summary"):
    st.write("Statistical Summary:", df.describe())

# Define categorical columns
categorical_columns = ['Gender', 'EmpNumber', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

# Encode categorical columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

#-------------visaulization-----------------
st.header("Visualization of the Dataset." , divider="rainbow")

#correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

st.write("Correlation Matrix Heatmap: To Calculate the correlation coefficients between all pairs of variables and visualize them as a heatmap. This allows you to quickly identify strong correlations (positive or negative) between variables.")

if st.checkbox("Correlation Heatmap"):
    st.write("Correlation Heatmap")
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


#prepare my data for testing and training
from sklearn.model_selection import train_test_split
x = df.drop('PerformanceRating' , axis=1)
y = df['PerformanceRating']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#create and fit the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(x_train,y_train)

#make predictions
y_pred = rfc.predict(x_test)

#check classification report
print('Classification Report', classification_report(y_test,y_pred))

#check accuracy
print('Accuracy', accuracy_score(y_test,y_pred))


#Boost my model
from sklearn.model_selection import GridSearchCV

#Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)

#Fit the GridSearchCV object to the data
grid_search.fit(x_train, y_train)

#Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Use the best estimator to make predictions
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(x_test)

#Evaluate the model
print('Best Parameters:', best_params)
print('Best Score:', best_score)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))

#Display model evaluation metrics in the app
st.subheader("Model Evaluation Metrics", divider="rainbow")
st.write(f"Best Parameters: {best_params}")
st.write(f"Best Score: {best_score}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

#Prediction section
st.sidebar.title("Enter Values to Predict Employee Performance")

# Dictionary containing options for categorical columns
categorical_options = {
    'Gender': ['Male', 'Female'],
    'EmpNumber': df['EmpNumber'].unique().tolist(),
    'EducationBackground': df['EducationBackground'].unique().tolist(),
    'MaritalStatus': ['Divorced','Married', 'Single'],
    'EmpDepartment': ['Data Science', 'Development', 'Finance', 'Human Resources', 'Research & Development', 'Sales'],
    'EmpJobRole': df['EmpJobRole'].unique().tolist(),
    'BusinessTravelFrequency': ['Non-Travel','Travel_Frequently', 'Travel_Rarely'],
    'OverTime': ['Yes', 'No'],
    'Attrition': ['Yes', 'No']
}       

#Create input fields for each feature
user_input = {}
for feature in x.columns:
    if feature in categorical_options:
        #Create a dropdown menu for categorical features
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", categorical_options[feature])
    else:
        #For numerical features, use text input
        user_input[feature] = st.sidebar.text_input(f"Enter {feature}", 0.0)

#Button to trigger the prediction
if st.sidebar.button("Predict"):
    #Convert user input into DataFrame
    user_input_df = pd.DataFrame([user_input])
    #Encode categorical variables
    for column in categorical_columns:
        user_input_df[column] = label_encoders[column].transform(user_input_df[column])
    #Predict using the trained model
    prediction = best_rf_model.predict(user_input_df)
    #Display the predicted result
    st.write('Predicted Employee Performance:', prediction[0])
    