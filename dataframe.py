import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from io import StringIO
import csv
import requests

def load_url(url):

    # Fetch the data
    response = requests.get(url)

    # Check if request was successful
    if response.status_code == 200:
        try:
            # Decode the content of the response
            content = response.content.decode()
            
            # Sniff the delimiter
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(content)

            dialect = sniffer.sniff(content.split('\n')[0])
            # Use the detected delimiter to split the content into a DataFrame
            if has_header:
                df = pd.read_csv(StringIO(content), delimiter=dialect.delimiter)
            else:
                df = pd.read_csv(StringIO(content), delimiter=dialect.delimiter, header=None)
                # Assign default column names if no header is present
                df.columns = [f'Column{i+1}' for i in range(df.shape[1])]

            # Convert numeric columns to numeric data types
            numeric_cols = df.columns[df.dtypes == object]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore') 
            
            return pre_analyse_df(df)   
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None, None, None, None, None   
    else:
        return None, None, None, None, None


def load_dataframe(uploaded_file):
    # Determine the file type and read the file into a DataFrame
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            
            delimiters = [',', ';', '\t', '|', ':', '\s+']  # List of potential delimiters
            try:
                lines = [uploaded_file.readline().decode().strip() for _ in range(5)]
                for delimiter in delimiters:
                    if any(delimiter in line for line in lines):
                        st.write(delimiter)
                        # Delimiter found in the lines, so use it to read the CSV
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(io.StringIO(uploaded_file.read().decode()), delimiter=delimiter)       

                # Convert numeric columns to numeric data types
                numeric_cols = df.columns[df.dtypes == object]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore') 
            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None
            
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("This file format is not supported. Please upload a CSV or Excel file.")
            return None, None, None, None, None
        
        return pre_analyse_df(df)  
    

def pre_analyse_df(df):
    
    num_rows, num_cols = df.shape
    st.sidebar.success(f"Original Data: {num_rows} rows and {num_cols} columns")

    st.header(":blue[Uploaded Data:]")
    st.dataframe(df)
    
    # Determine target column and determine if its numeric or categorical
    st.subheader(":blue[Step 1: Select Target Column i.e. value to be predicted by the analysis:]", divider='grey')
    target_column = [st.selectbox("Select the value to be predicted by the ML model:", index= None, options= df.columns, placeholder="Choose a target column (required)", label_visibility="collapsed")]

    if target_column[0] is not None:    

        if df[target_column[0]].dtype in ['int64', 'float64']:
            regression_type = 'numeric'
        else:
            regression_type = 'classification'      

        # Determine column types
        numeric_cols, categorical_cols = column_dtypes(df, target_column)
        
        # Filter data option
        df, filter_cols = filter_columns(df,target_column)   
        
        #Check to see if target_column has missing values
        if df[target_column[0]].isnull().any():
            df = target_missing_values(df, target_column,regression_type)     
        
        #Used columns = columns playing a role in input data
        if not isinstance(filter_cols, list):
            filter_cols = [filter_cols]
        used_cols = target_column + filter_cols
               
        # Identify categorical columns with large categorical unique values and drop if user confirms
        df, numeric_cols, categorical_cols = high_cardinality(df, numeric_cols, categorical_cols, used_cols, target_column)
 
        # Drop any user identified columns
        st.subheader(":blue[Step 3: Remove Additional Unrelated Columns:]", divider='grey')
        st.warning("Removing columns not related to the target column (optional) can improve accuracy and will not affect your original data. Examples of noise: duplicate columns, identification columns not related to value to be predicted, or non-independant columns. Select below to remove:")
        if st.toggle("Do not Remove any Additional Columns", key = 'keep_all_columns'):
            pass
        else:  
            drop_cols = st.multiselect("Select columns to drop by the ML model:", options= df.columns.difference(tuple(used_cols)), placeholder="Choose column(s) to remove from analysis (optional)", label_visibility="collapsed")

            if drop_cols:
                df, numeric_cols, categorical_cols = drop_columns(df, drop_cols, target_column)
                st.success(f"{len(drop_cols)} Column(s) dropped successfully.")

        # Identify columns with large missing data and drop if user confirms
        df, numeric_cols, categorical_cols = missing_data(df, numeric_cols, categorical_cols, used_cols, target_column)
        
            
        #Data is ready to send to machine learning model
        st.subheader(":blue[Step 4: Final Data for Analysis:]", divider='grey')
        num_rows, num_cols = df.shape
        st.sidebar.success(f"Data Used: {num_rows} rows and {num_cols} columns")
        st.dataframe(df)

        return df, numeric_cols, categorical_cols, target_column, regression_type
    else:
        return None, None, None, None, None
    
    
def target_missing_values(df, target_column, regression_type):
    
    missing_values_target = df[target_column[0]].isnull().sum()
    st.subheader(":blue[Step 2A: Missing Data in Target Column:]", divider='grey')
    if missing_values_target:
    
        
        st.error(f"{missing_values_target} of {len(df)} rows have missing values in the target column. Missing values affect predictive accuracy and are removed by default. If you prefer to replace missing values instead of deleting them, please select an option below:")
        
        if regression_type == 'numeric':
        
            mean_value = df[target_column[0]].mean()
            median_value = df[target_column[0]].median()
            mode_value = df[target_column[0]].mode().iloc[0]
                
            option = st.radio( "Choose an option to handle missing values:",
            ("Delete rows", f"Replace with mean: {round(mean_value,1)}",
            f"Replace with median: {round(median_value,1)}", f"Replace with mode: {round(mode_value,1)}"), index=None,label_visibility="collapsed", horizontal=True)

            if option == "Delete rows":
                df = df.dropna(subset=[target_column[0]])
                st.success("Rows deleted")
            elif option == "Replace with mean":
                df = df.fillna({target_column[0]: mean_value})
                st.success("Cells replaced with mean")
            elif option == "Replace with median":
                df = df.fillna({target_column[0]: median_value})
                st.success("Cells replaced with median")
            elif option == "Replace with mode":
                df = df.fillna({target_column[0]: mode_value})
                st.success("Cells replaced with mode")
            
        elif regression_type == 'classification':
            option = st.radio( "Choose an option to handle missing values:",
            ("Delete rows", "Replace with 'Missing'"), index=None,label_visibility="collapsed", horizontal=True)

            if option == "Delete rows":
                df = df.dropna(subset=[target_column[0]])
                st.success("Rows deleted")
            elif option == "Replace with 'Missing'":
                df = df.fillna({target_column[0]: 'Missing'})
                st.success("Cells replaced with 'Missing'")
    else:
        st.success("No missing data, continue to next step.")
    return df
    
    
def column_dtypes(df, target_column):

    # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(used_cols).tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(target_column).tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(target_column).tolist() 
    return numeric_cols, categorical_cols   

    
def filter_columns(df, target_column):

    st.subheader(":blue[Step 2: Filter Data:]", divider='grey')
    st.warning("Would you like to filter the data for the analysis? (optional)")
    if st.toggle("Do Not Filter Any Data", key="do_not_filter_data"):
        return df, None
    else:
        filter_cols = st.multiselect("Select columns to filter", options=df.columns, placeholder="Choose column(s) to filter for the analysis (optional)",label_visibility="collapsed")
   
        # Filtering options based on selected columns
        if filter_cols is not None:
            for idx, col in enumerate(filter_cols):
                st.write(f"Filtering for '**{col}**':")
                unique_values = df[col].dropna().unique()
                # Use the column name and index to create a unique key for each widget
                selected_value = st.selectbox(f"Select value for {col}", options=unique_values, key=f"{col}_value_{idx}")
                condition = st.selectbox("Condition", options=["==", ">", "<","<>"], key=f"{col}_condition_{idx}")
                
                # Apply filtering based on user selection
                if condition == "==":
                    df = df[df[col] == selected_value]
                elif condition == ">":
                    df = df[df[col] > selected_value]
                elif condition == "<":
                    df = df[df[col] < selected_value]
        return df, filter_cols
 
    
def drop_columns(df, drop_cols, target_column):
    df = df.drop(columns=drop_cols)
    
    # Update the columns lists after dropping
    numeric_cols, categorical_cols = column_dtypes(df, target_column)
    
    return df, numeric_cols, categorical_cols


def high_cardinality(df, numeric_cols, categorical_cols, used_cols, target_column):
    num_rows, num_cols = df.shape
    high_cardinality_threshold = max(50,round(num_rows/1000)*10)
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > high_cardinality_threshold and col not in used_cols]

     # Ask the user if these high cardinality columns should be dropped
    if high_cardinality_cols:
        st.subheader(f":blue[Step 2B: Assess Columns with >{high_cardinality_threshold} Unique Categories:]", divider='grey')
        st.warning("Columns with large # of unique categories may dilute performace and are excluded by default. To retain them, deselect them from the list below:")
        if st.toggle("Keep All Large Unique Category Column(s)", key="keep_all_high_cardinality_cols"):
            pass
        else:
            drop_high_cardinality_cols = st.multiselect(label = '',
                                                        options=high_cardinality_cols,
                                                        default=high_cardinality_cols, label_visibility="collapsed")  


            df, numeric_cols, categorical_cols = drop_columns(df, drop_high_cardinality_cols, target_column)
    else:
        st.subheader(f":blue[Step 2B: Columns with large # of Unique Categories:]", divider='grey')
        st.success("None found, continue to next step.")
    return df, numeric_cols, categorical_cols


def missing_data(df, numeric_cols, categorical_cols, used_cols, target_column):
    
    total_rows = len(df)
    high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0 or df[col].isnull().mean == 1 and col not in used_cols]
    st.subheader(f":blue[Step 3A: Additional Columns with Missing Data:]", divider='grey')
    # Ask the user if these columns should be dropped
    if high_missing_cols:
        
        num_missing_cols = len(high_missing_cols)
        percent_missing_cols = round((num_missing_cols / len(df.columns)) * 100, 2)
        num_missing_rows = df[high_missing_cols].isnull().any(axis=1).sum()
        percent_missing_rows = round((num_missing_rows / total_rows) * 100, 2)
        rows_lower_than_columns = 'rows' if percent_missing_cols >= percent_missing_rows else 'columns'
        
        
        st.warning(f"The following **'{num_missing_cols}'** columns have missing data affecting **'{num_missing_rows}'** rows. Missing values affect predictive accuracy; it is recommended {rows_lower_than_columns} be removed (default). If you prefer another option, please select below:") 
        
        drop_high_missing_cols = st.multiselect("Missing data columns",
                                                    options=high_missing_cols,
                                                    default=high_missing_cols, label_visibility="collapsed")
        
        option = st.radio("Choose an option to handle missing values:",
        ("Delete rows", "Delete columns", "Replace with '0' (may affect analysis)"), index=None,label_visibility="collapsed", horizontal=True)
                  
        if option == "Replace with '0' (may affect analysis)":
            for col in high_missing_cols:
                df[col].fillna(0, inplace=True)
            st.success(f"Missing cells replaced with '0'")
                
        elif option == "Delete rows":
            df.dropna(subset=high_missing_cols, inplace=True)
            st.success(f"{num_missing_rows} Rows deleted")

        else:
            df, numeric_cols, categorical_cols = drop_columns(df, drop_high_missing_cols, target_column)
            st.success(f"{len(high_missing_cols)} columns deleted")

        
        if df.empty:
            st.error("One or more selected columns contain missing data for all rows. Either remove the column in Step 4 or pick another selection in Step 5.")
            
    else:
        st.success("None found, continue to next step.")
    return df, numeric_cols, categorical_cols


def plot_columns(df, numeric_cols, categorical_cols, target_column):

    # Plotting graphs for numeric columns
    for col in numeric_cols:
        if col != target_column:
            fig, ax = plt.subplots(figsize=(2,1))
            sns.scatterplot(data=df, x=col, y=target_column, ax=ax)
            plt.title(f"{target_column} vs. {col}")
            st.pyplot(fig)
    
    # Plotting graphs for categorical columns
    for col in categorical_cols:
        if col != target_column:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, ax=ax)
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)