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
        numeric_cols = df.columns[df.dtypes != object]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric) 
        return pre_analyse_df(df)   
            
    else:
        e = response.status_code
        st.error(f"Error reading url data: {e}")
        return None, None, None, None, None   


def load_dataframe(uploaded_file):
    sniffer = csv.Sniffer()
    # Determine the file type and read the file into a DataFrame
    if uploaded_file is not None:
        if uploaded_file.name.endswith(('.csv', '.txt', '.data')):
            
            delimiters = [',', ';', '\t', '|', ':', '\s+']  # List of potential delimiters
            try:
                lines = [uploaded_file.readline().decode().strip() for _ in range(5)]
                has_header = sniffer.has_header('\n'.join(lines))
                dialect = sniffer.sniff('\n'.join(lines))
                # Reset file pointer
                uploaded_file.seek(0)
                for delimiter in delimiters:
                    if any(delimiter in line for line in lines):
                        df = pd.read_csv(io.StringIO(uploaded_file.read().decode()), delimiter=delimiter)
                        if not has_header:
                            df.columns = [f'Column{i+1}' for i in range(df.shape[1])]
                        break
                        # Delimiter found in the lines, so use it to read the CSV
                    else:
                        # If no delimiter found, use the dialect detected by the sniffer
                        df = pd.read_csv(uploaded_file, delimiter=dialect.delimiter, header=None if not has_header else 'infer')
                        if not has_header:
                            # Assign default column names if no header is present
                            df.columns = [f'Column{i+1}' for i in range(df.shape[1])]      
       

                # Convert numeric columns to numeric data types
                numeric_cols = df.columns[df.dtypes != object]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric) 
            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None
            
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            has_header = df.iloc[0].nunique() != len(df.columns)
            if has_header:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
                df.columns = [f'Column{i+1}' for i in range(df.shape[1])]
        else:
            st.error("This file format is not supported. Please upload a CSV or Excel file.")
            return None, None, None, None, None
        
        return pre_analyse_df(df)  
    
