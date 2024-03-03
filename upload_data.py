import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from io import StringIO
import csv
import requests
from dataframe import pre_analyse_df


def load_url(url):

    # Fetch the data
    response = requests.get(url)

    # Check if request was successful
    if response.status_code == 200:
       
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
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric) 

        return pre_analyse_df(df)   
            
    else:
        e = response.status_code
        st.error(f"Error reading url data: {e}")
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
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric) 
            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None
            
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("This file format is not supported. Please upload a CSV or Excel file.")
            return None, None, None, None, None
        
        return pre_analyse_df(df)  
    
