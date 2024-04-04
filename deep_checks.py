import streamlit as st
import pandas as pd
import deepchecks
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import *
from deepchecks.tabular.suites import data_integrity
import json 
import ppscore as pps
import plotly.graph_objects as go
from deepchecks.tabular.datasets.classification.phishing import load_data


n_to_show = 100000
n_samples = 100000

timeout = 120

def create_dataset(df, categorical_cols, target_column):
    dataset = Dataset(df, cat_features=categorical_cols, features=df.columns.difference(target_column).tolist(), label=target_column[0])
    return dataset

def pre_analyse_deep_checks(df, numeric_cols, categorical_cols, other_dtype_cols, target_column, regression_type):
    from dataframe import column_dtypes

    #Converting pandas dataframe to Deepchecks dataframe
    dataset = create_dataset(df, categorical_cols, target_column)

# Checks completed: 
    # Mixednulls (in dataframe.py), deepchecks version created but not needed,
    # Percent of Nulls (not needed), 
    # String Mismatch, created below and used
    # String Length Out of Bounds, created below but not needed
    # Feature Label Correlation, created using both ppscore and deepchecks, using ppscore as deepchecks uses a random sampling it seems providing different results each run
    # Is Single Value, created and used in dataframe.py - drop_constant_value_columns()
    # Feature Feature Correlation, created below and used
    # Columns Info, created but not using as deepchecks handles empty columns differently, using my version in dataframe.py
    # Conflicting Labels, created below and used
    # Data Duplicates, created below and used
    # Mixed Data Types, currently delete any other dtypes in dataframe.py, not using deepchecks at the moment
    # Special Characters, created below and used
    # Identifier Label Correlation, checks if identifies like index or date can predict Label using PPS score, not used corrently as all datetime and other columns are deleted
    # Outlier Sample Detection, created below and used
    # Class Imbalance, created below and used
    
    #Checks for MixedNulls, not used
    mixed_nulls(dataset)
    
    #Check for String MisMatch USA vs Usa
    string_mismatch(df, categorical_cols)
    
    #Check for String Length too long or too short than 'normal' length of column
    string_length(dataset)
    
    #Check for Features highly correlated with the Target column
    df, columns_edited_bool = feature_target_relation(dataset, df, target_column)
    numeric_cols, categorical_cols, other_dtype_cols = column_dtypes(df, target_column)
    dataset = dataset = create_dataset(df, categorical_cols, target_column)
    if columns_edited_bool is True:
        recheck_feature_target_relation(dataset, df, target_column)
    
    #Check for Features highly correlated with another Feature (age vs year of birth)
    df = feature_feature_relation(dataset, df, target_column)
    numeric_cols, categorical_cols, other_dtype_cols = column_dtypes(df, target_column)
    dataset = create_dataset(df, categorical_cols, target_column)
    
    #Check ColumnsInfo not using
    # st.write(ColumnsInfo().run(dataset).value)
    
    #Data Duplicates
    duplicates(dataset)
    
    #Special Characters
    special_characters(dataset)
    
    #Identifier Label Correlation
    # identifier_label_correlation(df, catagorical_cols, other_dtype_cols)
    
    #Outlier Detection
    outlier_detection(dataset)
    
    # Conflicting Labels check + Class Imbalance Check       
    if regression_type == 'classification' and df[target_column[0]].dtype in ['object', 'category','bool','boolean']:
        conflicting_labels(dataset)
        class_imbalance(dataset)
    
    return df
    
def mixed_nulls(df):
    
    null_datas = MixedNulls().run(df).to_json()
    null_data = json.loads(null_datas)
    columns_data = null_data['value']['columns']

    # Preparing data for Null DataFrame creation
    data = []
    for column_name, null_info in columns_data.items():
        for null_type, stats in null_info.items():
            row = {
                "Column Name": column_name,
                "Null Type": null_type,
                "Count": stats["count"],
                "Percent": stats["percent"] * 100  # Assuming you want to convert the fraction to percentage
            }
            data.append(row)

    # Creating the Nulls Summary DataFrame
    df_summary = pd.DataFrame(data)
    
    if df_summary.empty:
        pass
    else:
        return df_summary
    

def string_mismatch(df, catagorical_cols):
    string_mismatch_data = StringMismatch().run(df).to_json()
    string_mismatch_data = json.loads(string_mismatch_data)

    columns_data = string_mismatch_data['value']['columns']
    # Preparing data for DataFrame creation
    data = []
    st.subheader(":blue[String MisMatch:]", divider='grey')
    for column, details in columns_data.items():
        if isinstance(details, dict):
            for defect_type, defect_list in details.items():
                for defect in defect_list:
                    row = {
                        "Column Name": column,
                        "Defect Type": defect_type,
                        "Variant": defect["variant"],
                        "Count": defect["count"]["value"],
                        # "Percent": defect["percent"]["value"]
                    }
                    data.append(row)

    # Creating String MisMatched Summary DataFrame
    df_summary = pd.DataFrame(data)
    if df_summary.empty:
        st.success("No String MisMatch found. Proceed to the next step.")
        pass
    
    else:
        st.info("This table highlights columns with data that may represent the same category but are formatted differently (e.g., 'January' vs 'january'). To enhance accuracy, standardize these values in the original dataset and reload the data. Note: Without updates, these variants will remain treated as distinct categories.")
        st.write(df_summary)
        pass

    
def string_length(df):
    string_length_data = StringLengthOutOfBounds().run(df).to_json()
    string_length_data = json.loads(string_length_data)
    # st.write(string_length_data)
    # Preparing data for DataFrame creation
    data = []
    # Check if 'columns' key exists in 'value' dictionary
    if 'columns' in string_length_data['value']:
        for column, details in string_length_data['value']['columns'].items():
            # Assuming 'stats' and length keys exist
            min_length = details['stats']['min_length']['value']
            max_length = details['stats']['max_length']['value']

            row = {
                "Column Name": column,
                "Min Length": min_length,
                "Max Length": max_length
            }
            data.append(row)
    # Creating String Length Summary DataFrame
    df_summary = pd.DataFrame(data)
    if df_summary.empty:
        pass
    else:
        st.subheader(":blue[String Length; too long or too short:]", divider='grey')
        st.info("This section highlights columns where strings are significantly longer or shorter than typical values. Such discrepancies can indicate data quality issues or anomalies that may affect analysis or model performance and are provided for information. Note: if acceptable, no action is required.")
        st.write(df_summary)
        pass
    
    
def recheck_feature_target_relation(dataset, df, target_column):
    feature_relation_data = FeatureLabelCorrelation().run(dataset).to_json()
    feature_relation_data = json.loads(feature_relation_data)
    # Extracting feature names and their PPS scores
    data = [
        {"Feature Name": feature, "PPS Score": score}
        for feature, score in feature_relation_data["value"].items()
    ]
    
    df_feature_label_correlation = pd.DataFrame(data)
    filtered_df = df_feature_label_correlation[df_feature_label_correlation["PPS Score"] > 0]
    
    #Using ppscore library to calculate PPS scores, more time intensive as it uses all the data for every run - using this
    scores=[]
    for column in df.columns:
        if column != target_column[0]:
            score_dict = pps.score(df, column, target_column[0])
            scores.append({"Column": column, "PPS Score": score_dict["ppscore"]})
            
    pps_df = pd.DataFrame(scores)
    pps_df_sorted = pps_df.sort_values(by='PPS Score', ascending=False)
    pps_df_sorted = pps_df_sorted[pps_df_sorted["PPS Score"] > 0.2] #Threshold picked as 0.2
    if pps_df_sorted.empty:
        st.success("No feature-target relation > a PPS score of 0.2 found. Proceed to the next step.")
    else:
        st.write(pps_df_sorted)
 
 
def feature_target_relation(dataset, df, target_column):
    #Using deepchecks library to calculate PPS scores, there seems to be a random sample assumption somewhere, provides slighly different results on every run
    feature_relation_data = FeatureLabelCorrelation().run(dataset).to_json()
    feature_relation_data = json.loads(feature_relation_data)
    # Extracting feature names and their PPS scores
    data = [
        {"Feature Name": feature, "PPS Score": score}
        for feature, score in feature_relation_data["value"].items()
    ]
    
    df_feature_label_correlation = pd.DataFrame(data)
    filtered_df = df_feature_label_correlation[df_feature_label_correlation["PPS Score"] > 0]
    
    #Using ppscore library to calculate PPS scores, more time intensive as it uses all the data for every run - using this
    scores=[]
    for column in df.columns:
        if column != target_column[0]:
            score_dict = pps.score(df, column, target_column[0])
            scores.append({"Column": column, "PPS Score": score_dict["ppscore"]})
            
    pps_df = pd.DataFrame(scores)
    pps_df_sorted = pps_df.sort_values(by='PPS Score', ascending=False)
    pps_df_sorted = pps_df_sorted[pps_df_sorted["PPS Score"] > 0.2] #Threshold picked as 0.2
    st.subheader(":blue[Feature to Target Column Relation:]", divider='grey')
    if pps_df_sorted.empty:
        st.info("No feature-target relation > a PPS score of 0.2 found. Proceed to the next step.")
        return df, False
    else:
        st.info("This table shows how closely related each feature is to the target variable. High scores may indicate that a feature is too predictive of the outcome (i.e. 'data leakage'), leading to misleadingly good results. To avoid this, you might want to exclude any features with high scores from your analysis, unless they are justifiably independent. For moderate or low scores, no adjustments are necessary.")
        drop_cols = st.multiselect("Select columns to drop by the ML model:", options= pps_df_sorted['Column'].unique(), placeholder="Choose column(s) to remove from analysis (optional)", label_visibility="collapsed", key = 'feature_target_drop')
        st.write(pps_df_sorted)
        if drop_cols:
            df = df.drop(columns=drop_cols)
            st.success(f'Columns dropped from analysis: {format(drop_cols)}. Updated table below:')
            return df, True
        else:
            return df, False


def feature_feature_relation(dataset, df, target_column):
    #Using deepchecks library to calculate feature to feature relation to identify if redundant columns exist. Shows top 10 columns
    feature_feature_relation_data = FeatureFeatureCorrelation().run(dataset= dataset, feature_importance_timeout=timeout)
    
    st.subheader(":blue[Feature to Feature Relation Summary:]", divider='grey')
    
    st.info("""This chart visualizes the correlation between feature pairs. A high correlation score suggests redundancy or duplication between features, for example, 'age' and 'year of birth' often contain overlapping information. Removing or combining these highly correlated features can enhance model efficiency and minimize potential biases. If no features show high correlation scores or you're certain there's no redundancy among the columns, proceed to the next step.""")

    drop_cols = st.multiselect("Select columns to drop by the ML model:", options=df.columns.difference(target_column), placeholder="Choose column(s) to remove from analysis (optional)", label_visibility="collapsed", key = 'feature_feature_drop')
    st.plotly_chart(feature_feature_relation_data.display[0])
    st.write(feature_feature_relation_data.display[1])
    # st.write(feature_feature_relation_data.value)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        st.success('Columns dropped from analysis: {}'.format(drop_cols))
        return df
    else:
        return df


def conflicting_labels(dataset):

    # phishing_dataframe = load_data(as_train_test=False, data_format='Dataframe')
    # phishing_dataset = Dataset(phishing_dataframe, label='target', features=['urlLength', 'numDigits', 'numParams', 'num_%20', 'num_@', 'bodyLength', 'numTitles', 'numImages', 'numLinks', 'specialChars'])
    st.subheader(":blue[Conflicting Labels Summary:]", divider='grey')  
    conflicting_labels_data = ConflictingLabels(n_to_show=n_to_show).run(dataset).value
    # conflicting_labels_data = ConflictingLabels().run(phishing_dataset).value
    if conflicting_labels_data['percent_of_conflicting_samples']>0:      
        st.info("Each row in the table below highlights row index numbers that share identical data across all features but differ in the Target column. These inconsistencies, showing the same inputs with conflicting outcomes, can confuse the model. Addressing these discrepancies in the original dataset by either removing or correcting them will enhance the model's prediction accuracy and reliability.")
        # st.write(conflicting_labels_data['percent_of_conflicting_samples'])
        df_conflicting_labels_data = pd.DataFrame(conflicting_labels_data['samples_indices'])
        st.write(df_conflicting_labels_data)
    else:
        st.info("No conflicting label data found. Proceed to the next step.")
        
        
def duplicates(dataset):
    duplicates_data = DataDuplicates(n_to_show=n_to_show).run(dataset)
    st.subheader(":blue[Duplicate Data Summary:]", divider='grey')  
    if duplicates_data.value>0:              
        st.info("Each row in the table below highlights row index numbers with identical data across all features. Duplicate samples increase the weight the model gives to those samples affecting accuracy and reliability. If these duplicates are intentional, you may disregard this notice. Otherwise, address them by removal or correction in the original dataset.")
        df_duplicate_data = pd.DataFrame(duplicates_data.display[2])
        st.write(df_duplicate_data)
    else:
        st.success("No duplicate data found. Proceed to the next step.")


def special_characters(dataset):
    st.subheader(":blue[Special Characters Summary:]", divider='grey')
    # data = {'col1': [' ', '!', '"', '#', '$', '%', '&', '\'','(', ')',
    #              '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
    #              '>', '?', '@', '[', ']', '\\', '^', '_', '`', '{',
    #              '}', '|', '~', '\n'],
    #     'col2':['v', 'v', 'v', 'v4', 'v5', 'v6', 'v7', 'v8','v9','v10',
    #              '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
    #              '>', '?', '@', '[', ']', '\\', '^', '_', '`', '{',
    #              '}', '|', '~', '\n'],
    #     'col3': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,'???#',1,1,1,1,1,1,1,1,1,1,1],
    #     'col4': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     'col5': ['valid1','valid2','valid3','valid4','valid5','valid6','valid7',
    #              'valid8','valid9','valid10','valid11','valid12',
    #             'valid13','valid14','inval!d15','valid16','valid17','valid18',
    #              'valid19','valid20','valid21','valid22','valid23','valid24','valid25',
    #             'valid26', 'valid27','valid28','valid29','valid30','valid31','32','33','34']}

    # dataset = pd.DataFrame(data=data)
    special_characters_data = SpecialCharacters(n_top_columns=n_to_show).run(dataset).value
    df_special_characters_data = pd.DataFrame([special_characters_data])
    df_special_characters_data = df_special_characters_data.transpose()
    df_special_characters_data.columns = ['% of Special Characters']
    df_special_characters_data['% of Special Characters'] = df_special_characters_data['% of Special Characters']*100

    if df_special_characters_data['% of Special Characters'].gt(0).any():
        st.info("This table shows the percentage of entries with special characters in each column. If the presence of speical characters is intential, disregard this notice. Otherwise, address them by removal or correction in the original dataset.")
        st.dataframe(df_special_characters_data)
    else:
        st.success("No special characters found. Proceed to the next step.")

#need to run this properly when other_dtype_cols are included in the analysis, currently not functional
def identifier_label_correlation(df, catagorical_cols, other_dtype_cols):
    st.subheader(":blue[Identifier Label Correlation Summary:]", divider='grey')
    dataset = Dataset(df, label='target', index_name='id', datetime_name='datetime',cat_features=catagorical_cols)
    identifier_label_correlation_data = IdentifierLabelCorrelation(n_to_show=n_to_show).run(dataset)


def outlier_detection(dataset):
    st.subheader(":blue[Outlier Summary:]", divider='grey')
    
    try:
        outlier_detection_data = OutlierSampleDetection(n_to_show=n_to_show, n_samples=n_samples, timeout=timeout).run(dataset)
        df_outlier_detection_data = pd.DataFrame(outlier_detection_data.display[1]) 
        df_filtered = df_outlier_detection_data.loc[df_outlier_detection_data['Outlier Probability Score'] > 0.8]
        
        if not df_filtered.empty:
            st.info("This table shows entries that have a > 80% probability of being an outlier based on the LoOp algorithm. If outliers are important and should stay in the dataset, disregard this notice. Otherwise address them by verification, removal or correction in the original dataset and reload the data.")
            st.dataframe(df_filtered)
        else:
            st.success("No outliers found. Proceed to the next step.")
    except:
            st.error("Outlier detection failed either due to data entered or it took longer than 2 mins.")
            
def class_imbalance(dataset):
    st.subheader(":blue[Class Imbalance Summary:]", divider='grey')
    class_imbalance_data = ClassImbalance(n_to_show=n_to_show).run(dataset)    
    st.info("The figure below shows distribution of classes, highlighting potential class imbalances. This occurs when the frequency of one class far exceeds another, possibly skewing predictions. If the imbalance reflects real-world scenarios, like more legitimate than fraudulent bank transactions, disregard this notice. Otherwise, consider balancing the classes by adding more data to the underrepresented class.")
    st.write(class_imbalance_data.display[0])