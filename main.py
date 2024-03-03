import streamlit as st
from randomforest import random_forest
from upload_data import load_dataframe, load_url
from generic_layout import generic_main, generic_footer


# Streamlit interface
generic_main()
st.sidebar.title('No-Code Machine Learning Prediction Model for Non-Techies')

st.subheader(":blue[How to Use:]")
# Purpose
st.markdown(""":blue[Purpose of this App:] The aim of this app is to help users identify trend in data worth exploring without coding or machine learning knowledge/experience. It should be noted that this is a basic machine learning model. Further analysis is needed to address bias, overfitting, underfitting, and ensure dataset is balanced for classification problems.""")


# Step 1: Clean Data
st.markdown(""" :blue[Clean Data:] Remove errors, duplicates, and unnecessary columns. Handle missing values. Ensure each column contains either numbers or text. Mixing text and numbers will cause errors. For classification problems, ensure your target column contains balanced classes.""")

# Step 2: Upload CSV
st.markdown(""":blue[Upload Data:] Use the file/url uploader on sidebar. Select the target (prediction) column. """)

# Patience
st.markdown(""":blue[Please be patient. Analyzing data > 5 columns and 1000 rows can take > 5 mins... We are working on improving this.]""")


# File uploader allows user to add their own Excel
uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['csv','xls','xlsx'])

# Input field for URL
url = st.sidebar.text_input("Or Provide Direct Download URL", value=None, placeholder="Enter URL and press Enter", key='url')
st.sidebar.write("Example url: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv")
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
    
    

#         #gradeint_descent(df_filtered)
#         #tensor_flow_train_model(df_filtered, numeric_cols, categorical_cols, target_column)


generic_footer()



