import streamlit as st
from randomforest import random_forest, random_forest_optimizer
from dataframe import load_dataframe, load_url
from gradientdecent import gradeint_descent
from generic_layout import generic_main, generic_footer



# Streamlit interface
generic_main()
st.sidebar.title('No-Code Machine Learning Prediction Model for Non-Techies')

st.subheader(":blue[How to Use:]")
# Purpose
st.markdown(""":blue[Purpose of this App:] The aim of this app is to help users identify trends in data worth exploring without coding or machine learning knowledge/experience. It should be noted that this is a basic machine learning model. Further analysis is needed to address bias, overfitting, underfitting, and ensure dataset is balanced for classification problems.""")


# Step 1: Clean Data
st.markdown(""" :blue[Clean Data:] Remove errors, duplicates, and unnecessary columns. Handle missing values. For classification problem, ensure your target column contains balanced classes.""")

# Step 2: Upload CSV
st.markdown(""":blue[Upload Data:] Use the file/url uploader on sidebar. Select the target (prediction) column. """)

# File uploader allows user to add their own Excel
uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['csv','xls','xlsx'])

# Input field for URL
url = st.sidebar.text_input("Or Provide Direct Download URL", value=None, placeholder="Enter URL and press Enter", key='url')

if uploaded_file is not None:
    df,  numeric_cols, categorical_cols, target_column, regression_type = load_dataframe(uploaded_file)
    if df is not None: 
       random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)
        
elif url is not None:
    
    df,  numeric_cols, categorical_cols, target_column, regression_type = load_url(url)
    if df is not None: 
        random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)

    
else:
    st.info('Please upload a file or enter a url to begin.')
    
# random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)
    
# if target_column is not None:
#         #gradeint_descent(df_filtered)
#         # Preprocess and train the model
#         # if st.button('Train Model'):
#         random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)
#         #random_forest_optimizer(df, numeric_cols, categorical_cols, target_column)
#         #tensor_flow_train_model(df_filtered, numeric_cols, categorical_cols, target_column)


generic_footer()



