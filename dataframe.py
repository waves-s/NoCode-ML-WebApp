import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from io import StringIO
import csv
import requests   

def pre_analyse_df(df):
    
    num_rows, num_cols = df.shape
    st.sidebar.success(f"Original Data: {num_rows} rows and {num_cols} columns")

    st.header(":blue[Original Uploaded Data:]")
    st.dataframe(df)
    
    #Delete text-numeric columns
    df = drop_text_numeric_columns(df)
    
    #Delete constant value columns
    df = drop_constant_value_columns(df)
    
    # Determine target column and determine if its numeric or categorical
    st.subheader(":blue[Step 3: Select Target Column i.e. value to be predicted by the analysis:]", divider='grey')
    target_column = [st.selectbox("Select the value to be predicted by the ML model:", index= None, options= df.columns, placeholder="Choose a target column (required)", label_visibility="collapsed")]
    target_type = [st.selectbox("Select the regression type to be predicted by the ML model:", index= None, options=['Continuous','Categories'], placeholder="Is the target column 'continuous' e.g. age, temperature or discrete 'categories' e.g. types of flowers, 0 or 1 (True or False) etc", label_visibility="collapsed")]
    
    if target_column[0] is not None and target_type[0] is not None:

        if df[target_column[0]].dtype in ['object'] and target_type[0] == 'Continuous':
            st.error("Target column is non-numeric and can not be assessed as continuous. Please select categorical.")
            return None, None, None, None, None
 
        else:
            
            if target_type[0] == 'Continuous':
                regression_type = 'numeric'
            else:
                regression_type = 'classification'
            return data_clean_up(df, target_column, regression_type)
    else:
        return None, None, None, None, None
            
        
def data_clean_up(df, target_column, regression_type):
    
    # Determine column types
    numeric_cols, categorical_cols = column_dtypes(df, target_column)
    
    #Check to see if target_column has missing values
    # if df[target_column[0]].isnull().any():
    df = target_missing_values(df, target_column,regression_type)  
    
    # Filter data option
    df, filter_cols = filter_columns(df,target_column)      
    
    #Used columns = columns playing a role in input data
    if not isinstance(filter_cols, list):
        filter_cols = [filter_cols]
    used_cols = target_column + filter_cols
        
    # Identify categorical columns with large categorical unique values and drop if user confirms
    df, numeric_cols, categorical_cols = high_cardinality(df, numeric_cols, categorical_cols, used_cols, target_column)
    
    # Identify columns with large missing data and drop if user confirms
    df, numeric_cols, categorical_cols = missing_data(df, numeric_cols, categorical_cols, used_cols, target_column)

    # Drop any user identified columns
    st.subheader(":blue[Step 8: Remove Additional Unrelated Columns:]", divider='grey')
    st.warning("Removing columns not related to the target column (optional) can improve accuracy. Examples of noise: duplicate columns, identification columns not related to value to be predicted, or non-independant columns. Select below to remove:")
    if st.toggle("Do not Remove any Additional Columns", key = 'keep_all_columns'):
        pass
    else:  
        drop_cols = st.multiselect("Select columns to drop by the ML model:", options= df.columns.difference(tuple(used_cols)), placeholder="Choose column(s) to remove from analysis (optional)", label_visibility="collapsed")

        if drop_cols:
            df, numeric_cols, categorical_cols = drop_columns(df, drop_cols, target_column)
            st.success(f"{len(drop_cols)} Column(s) dropped successfully.")
        
    #Data is ready to send to machine learning model
    st.subheader(":blue[Step 9: Final Data for Analysis:]", divider='grey')
    num_rows, num_cols = df.shape
    st.sidebar.success(f"Data Used: {num_rows} rows and {num_cols} columns")
    st.dataframe(df)

    return df, numeric_cols, categorical_cols, target_column, regression_type

    
    
def target_missing_values(df, target_column, regression_type):
    
    total_rows, total_columns = df.shape
    missing_rows_target = df[target_column[0]].isnull().sum()
    st.subheader(":blue[Step 4: Missing Data in Target Column:]", divider='grey')   
    
    if missing_rows_target:
        
        # num_missing_rows = df[missing_rows_target].isnull().any(axis=1).sum()
        num_missing_rows = missing_rows_target
        # percent_missing_rows = round((num_missing_rows / total_rows) * 100, 2)
        
        st.error(f"Target column has missing data affecting **'{num_missing_rows}'** rows of **'{total_rows}'** total rows. Missing values affect predictive accuracy and are removed by default. If you prefer another option, please select below:")

        if regression_type == 'numeric':
        
            mean_value = df[target_column[0]].mean()
            mean_value_text = f"Replace with mean: {mean_value:.1f}"
            median_value = df[target_column[0]].median()
            median_value_text = f"Replace with median: {median_value:.1f}"
            mode_value = df[target_column[0]].mode().iloc[0]
            mode_value_text = f"Replace with mode: {mode_value:.1f}"
                
            option = st.radio( "Choose an option to handle missing values:", ("Delete rows", mean_value_text,median_value_text, mode_value_text), index=None,label_visibility="collapsed", horizontal=True)

            if option == mean_value_text:
                df = df.fillna({target_column[0]: mean_value})
                st.success(f"{num_missing_rows} row(s) replaced with mean")
            elif option == median_value_text:
                df = df.fillna({target_column[0]: median_value})
                st.success(f" {num_missing_rows} row(s) replaced with median")
            elif option == mode_value_text:
                df = df.fillna({target_column[0]: mode_value})
                st.success(f" {num_missing_rows} row(s) replaced with mode")
            else:
                df = df.dropna(subset=[target_column[0]])
                st.success(f" {num_missing_rows} row(s) deleted")
            
        elif regression_type == 'classification':
            option = st.radio( "Choose an option to handle missing values:",
            ("Delete rows", "Replace with 'Missing'"), index=None,label_visibility="collapsed", horizontal=True)

            if option == "Replace with 'Missing'":
                df = df.fillna({target_column[0]: 'Missing'})
                st.success(f" {num_missing_rows} row(s) replaced with 'Missing'")
            else:
                df = df.dropna(subset=[target_column[0]])
                st.success(f"{num_missing_rows} Row(s) deleted")
    else:
        st.success("No missing data in Target column.")
    return df
    
    
def column_dtypes(df, target_column):

    # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(used_cols).tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(target_column).tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(target_column).tolist() 
    return numeric_cols, categorical_cols   

    
def filter_columns(df, target_column):

    st.subheader(":blue[Step 5: Filter Data:]", divider='grey')
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


def drop_text_numeric_columns(df):
    # Find columns with text-numeric data
    text_numeric_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].astype(str).str.contains(r'^-?\d*\.?\d+$').any()]
    st.subheader(":blue[Step 1: Text-Numeric Columns]", divider='grey')
    # Drop columns with text-numeric data
    if text_numeric_cols:
        
        df =  df.drop(text_numeric_cols, axis=1)
        st.success(f"Model can only analyise numeric-only or text-only data. {len(text_numeric_cols)} column(s) with text-numeric data have been deleted: **{text_numeric_cols}**")
    else:
        st.success("No columns with text-numeric data found.")
    return df


def drop_constant_value_columns(df):
    constant_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    st.subheader(":blue[Step 2: Removing Constant Value Columns]", divider='grey')
    # Drop columns with constant values
    if constant_value_cols:
        
        df = df.drop(constant_value_cols, axis=1)
        st.success(f"Columns with only 1 unique value does not contribute to the analysis. {len(constant_value_cols)} column(s) with constant values have been deleted: **{constant_value_cols}**")
    else:
        st.success("No columns with constant values found.")

    return df


def high_cardinality(df, numeric_cols, categorical_cols, used_cols, target_column):
    num_rows, num_cols = df.shape
    high_cardinality_threshold = max(50,round(num_rows/1000)*10)
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > high_cardinality_threshold and col not in used_cols]
    st.subheader(f":blue[Step 6: Columns with >{high_cardinality_threshold} Unique Categories:]", divider='grey')
    
    # Ask the user if these high cardinality columns should be dropped
    if high_cardinality_cols:
        
        st.warning("Columns with large # of unique categories may dilute performace and are excluded by default. To retain them, deselect them from the list below:")
        if st.toggle("Keep All Large Unique Category Column(s)", key="keep_all_high_cardinality_cols"):
            pass
        else:
            drop_high_cardinality_cols = st.multiselect(label = '',
                                                        options=high_cardinality_cols,
                                                        default=high_cardinality_cols, label_visibility="collapsed")  


            df, numeric_cols, categorical_cols = drop_columns(df, drop_high_cardinality_cols, target_column)
    else:
        st.success("No columns with large # of unique categories found.")
    return df, numeric_cols, categorical_cols


def missing_data(df, numeric_cols, categorical_cols, used_cols, target_column):
    
    total_rows, total_columns = df.shape
    high_missing_cols = [col for col in df.columns.difference(used_cols) if df[col].isnull().mean() > 0 or df[col].isnull().mean == 1 and col not in used_cols]
    st.subheader(f":blue[Step 7: Additional Columns with Missing Data:]", divider='grey')
    # Ask the user if these columns should be dropped
    if high_missing_cols:
        
        num_missing_cols = len(high_missing_cols)
        percent_missing_cols = round((num_missing_cols / total_columns) * 100, 2)
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
        st.success("No columns found with missing data.")
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
            
            
            
            # if df[target_column[0]].dtype in ['int64', 'float64']:
            #     regression_type = 'numeric'
            # else:
            #     regression_type = 'classification'    