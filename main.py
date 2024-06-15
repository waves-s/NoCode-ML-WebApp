import streamlit as st
from randomforest import random_forest
from upload_data import load_dataframe, load_url
from generic_layout import generic_main, generic_footer
from dataframe import pre_analyse_df


# Streamlit interface
generic_main()
st.sidebar.title('No-Code Machine Learning Prediction Model for Non-Techies')

st.subheader(":blue[How to Use:]")
# Purpose
st.markdown(""":blue[Purpose of this App:] Discover trends in tabular data (numeric or categorical data) without coding or machine learning knowledge. Note: This is a basic machine learning model, further analysis to evaluate bias, overfitting, underfitting and balanced data for classification problems should be conducted before deployment.""")


# Step 1: Clean Data
st.markdown("""
:blue[Clean Data:] Note the following for input file upload:
- Data headers should be in row 1  
- CSV/Excel file should have only a single sheet  
- No formatting (e.g. filters), charts, figures  
- Ensure good data quality i.e. remove errors, duplicates, and handle missing values.  
- Columns must contain either numbers or text. Text-numeric columns will be deleted  
- For classification analysis, ensure the target column exhibits balanced classes i.e. each class is represented proportionally to real-world prevalence. e.g. if population is evenly split between genders or if certain classes are rare like people over 6 feet tall, then the dataset should reflect that.
""")

# Step 2: Upload CSV
st.markdown(""":blue[Upload Data:] Use the file/url uploader on sidebar. Select the target (prediction) column. """)

# Patience
st.markdown(""":blue[Please be patient. Analyzing data > 5 columns and 1000 rows can take > 5 mins... We are working on improving this.]""")


# File uploader allows user to add their own Excel
uploaded_file = st.sidebar.file_uploader("Upload your input file (single sheet only)", type=['csv','xls','xlsx'])

# Input field for URL
url = st.sidebar.text_input("Or Provide Direct Download URL", value=None, placeholder="Enter URL and press Enter", key='url')
st.sidebar.write("Example url: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
if uploaded_file is not None:
    df = load_dataframe(uploaded_file)
    df, numeric_cols, categorical_cols, target_column, regression_type = pre_analyse_df(df)  

    if df is not None: 
       random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)
        
elif url is not None:
    try:
        df,  numeric_cols, categorical_cols, target_column, regression_type = load_url(url)
        if df is not None: 
            random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info('Please upload a file or enter a url to begin.')
    
    

#         #gradeint_descent(df_filtered)
#         #tensor_flow_train_model(df_filtered, numeric_cols, categorical_cols, target_column)


generic_footer()



