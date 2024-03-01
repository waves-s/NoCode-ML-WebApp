# RandomForest is an ensemble method (a method that argues that if multiple decision trees are created, we get a better result) of decision trees. The decisions made by the model can be traced back through the trees, offering better interpretability than deep learning models.
# Can artifically add x% to generate more conservative results y_pred = y_pred + 0.05

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# Define a custom transformer to apply LabelEncoder
def label_encode_column(column):
    encoder = LabelEncoder()
    encoded_column = encoder.fit_transform(column)
    # Reshape the 1D array to 2D array
    return encoded_column.reshape(-1, 1)


def rmse_accuracy(regression_type, y_test, y_pred):
    if regression_type == 'numeric':
        rmse = root_mean_squared_error(y_test, y_pred)
        return rmse
        
    elif regression_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


def random_forest_optimizer(df, numeric_cols, classification_cols, target_column,regression_type):
    try:        
        best_rmse = 100.0
        # Split the data into training and test sets
        X = df[numeric_cols + classification_cols]
        y = df[target_column] 

        if regression_type == 'classification':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            best_rmse = -100.0

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
            
        #No Imputer is used, Random Forest inherently ignores missing values
        #Currently no Feature Engineering is performed
        
        # Preprocessing for classification data
        # classification_transformer = Pipeline(steps=[
        #     ('encoder', FunctionTransformer(label_encode_column, validate=False))])
       classification_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))])
        
        # Feature Scaling - Use both StandardScaler and MinMaxScaler to determine best fit
        scalers = [('StandardScaler', StandardScaler()), ('MinMaxScaler', MinMaxScaler())]

        all_parameters = {}
        index = 0

        # Loop through each scaler
        for scaler_name, scaler in scalers:
            numeric_transformer = Pipeline(steps=[
                ('scaler', scaler)  
            ])
            
            # Bundle preprocessing for numeric and categorical data
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', classification_transformer, classification_cols)
                ])
           
            #Identifying model parameters for optimization
            #Criterion using default of mean sqared error, change if need to be more robust to outliers
            n_estimators = [50,100,150,200] #Default is 100
            #Max_depth default is none, reduce if overfitting
            min_sample_split = [2,5,10] #Default is 2
            max_features = ['auto', 'sqrt'] #Not emcompassing log2 or interget/float values
            #Bootstrap is defaulted to true, false would use whole dataset to build each tree which can reduce bias, but true reduces overfitting
            #oob_score default to false, make this true if looking for a quick unbiased estimate without setting aside a validation set, beneficial where data is scarse
            #max_leaf_nodes default to none, change to control overfitting
            min_impurity = [0, 0.01, 0.1] #min_impurity_decrease default 0, higher values can prevent overly complex trees
            
            for estimators in n_estimators:
                for impurity in min_impurity:
                    # Define the model
                    if regression_type == 'numeric':
                        model = RandomForestRegressor(n_estimators=estimators, random_state=42, min_impurity_decrease=impurity)
                    elif regression_type == 'classification':
                        #no min_impurity_decrease parameter for Classifier
                        model = RandomForestClassifier(n_estimators=estimators, random_state=42,)
                    
                    # Bundle preprocessing and modeling code in a pipeline
                    clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)])
                    
                    # Preprocessing of training data, fit model 
                    clf.fit(X_train, y_train)
                    
                    # Get predictions
                    y_pred = clf.predict(X_test)
                    if isinstance(y, np.ndarray):
                        if y.dtype == 'int64' or y.dtype == 'int32':
                            y_pred = y_pred.round().astype(int)
                    elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                        if (y.dtypes == 'int64').all() or (y.dtypes == 'int32').all():
                            y_pred = y_pred.round().astype(int)


                    #Calculate RMSE/Accuracy
                    current_metric = rmse_accuracy(regression_type, y_test, y_pred) 
                
                    if regression_type == 'classification':
                        y_pred = label_encoder.inverse_transform(y_pred)
                        if current_metric >= best_rmse:
                            best_rmse = current_metric
                    elif regression_type == 'numeric':
                        if current_metric <= best_rmse:
                            best_rmse = current_metric
                     
                    # if current_metric <= best_rmse:
                    #     best_rmse = current_metric
                    
                    # Store RMSE/Accuracy and corresponding parameters
                    parameters = {'RMSE/Accuracy': current_metric, 'scaler': scaler_name, 'n_estimators': estimators, 'min_impurity': impurity}
                    all_parameters[index] = parameters
                    index += 1      
                                      
        # Output the best RMSE and corresponding parameters
        for key, value in all_parameters.items():
            if value['RMSE/Accuracy'] == best_rmse:
                best_parameter = all_parameters[key]
            else:
                pass

        # Convert the dictionary to a DataFrame
        df_params = pd.DataFrame.from_dict(all_parameters, orient='index')
        df_params.set_index('RMSE/Accuracy', inplace=True)  # Reset the index to turn it into a column
        
        return best_parameter, df_params
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
      
def random_forest(df, numeric_cols, classification_cols, target_column, regression_type):

    try:
        # Split the data into training and test sets
        X = df[numeric_cols + classification_cols]  
        y = df[target_column]        
        
        # Define the model 
        if regression_type == 'numeric':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif regression_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing for numeric data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])
        
        # Preprocessing for classification data
        # classification_transformer = Pipeline(steps=[
        #     ('encoder', FunctionTransformer(label_encode_column, validate = False))])
        classification_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))])

        # Bundle preprocessing for numeric and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', classification_transformer, classification_cols)
            ])

        # processed_data = preprocessor.fit_transform(df)
       
        # Bundle preprocessing and modeling code in a pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)])
           
        # Preprocessing of training data, fit model 
        clf.fit(X_train, y_train)

        # Get predictions
        y_pred = clf.predict(X_test)

        if isinstance(y, np.ndarray):
            if y.dtype == 'int64' or y.dtype == 'int32':
                y_pred = y_pred.round().astype(int)
        elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            if (y.dtypes == 'int64').all() or (y.dtypes == 'int32').all():
                y_pred = y_pred.round().astype(int)
            
        # Calculate RMSE/Accuracy
        current_metric = rmse_accuracy(regression_type, y_test, y_pred)

        if regression_type == 'classification':
            y_pred = label_encoder.inverse_transform(y_pred)

        st.subheader(":blue[Random Forest Results:]", divider='grey')
        wording = "Root Mean Squared Error" if regression_type == 'numeric' else "Accuracy"
        st.write(f":blue[1st Default {wording}: {current_metric}]")
            
        if st.toggle("Run Optimization for Model", key="optimize_randomforest"):
            best_parameters, df_params = random_forest_optimizer(df, numeric_cols, classification_cols, target_column, regression_type)
            scaler_name = best_parameters['scaler']
            estimators = best_parameters['n_estimators']
            impurity = best_parameters['min_impurity']
            optimized_rsme_accuracy = best_parameters['RMSE/Accuracy']

            if current_metric == optimized_rsme_accuracy:
                st.success(f"Best {wording} Achieved with default parameters")
            
            else:
                st.write(f":blue[Best Optimized {wording}: {optimized_rsme_accuracy}]")
                st.write(f":blue[Parameters: {optimized_rsme_accuracy}]")
                st.dataframe(df_params.sort_values(by='RMSE/Accuracy', ascending=True))
    
            
                if regression_type == 'numeric':
                    model = RandomForestRegressor(n_estimators=estimators, random_state=42, min_impurity_decrease=impurity)
                
                elif regression_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                
                # Bundle preprocessing and modeling code in a pipeline
                clf = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

                # Preprocessing of training data, fit model 
                clf.fit(X_train, y_train)
        
                # Get predictions
                y_pred = clf.predict(X_test)
                if isinstance(y, np.ndarray):
                    if y.dtype == 'int64' or y.dtype == 'int32':
                        y_pred = y_pred.round().astype(int)
                elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                    if (y.dtypes == 'int64').all() or (y.dtypes == 'int32').all():
                        y_pred = y_pred.round().astype(int)
                        
                #Calculate RMSE/Accuracy
                current_metric = rmse_accuracy(regression_type, y_test, y_pred) 
            
                # Calculate differences: positive values mean the prediction is higher than the actual (conservative)
                if regression_type == 'numeric':
                    differences = y_pred - y_test.values.flatten()
                    # Calculate the conservativeness metrics
                    conservative_predictions = (differences > 0).sum()
                    total_predictions = len(differences)
                    percentage_conservative = (conservative_predictions / total_predictions) * 100
                    st.write(f':blue[Percentage of Conservative Predictions: {percentage_conservative:.2f}%]')

                    # Identify non-conservative (lower than actual) predictions and their percentage
                    non_conservative_predictions = (differences <= 0).sum()
                    percentage_non_conservative = (non_conservative_predictions / total_predictions) * 100

                    st.write(f':blue[Percentage of Non-Conservative Predictions: {percentage_non_conservative:.2f}%]')
    
                    # Create a DataFrame for comparing actual vs. predicted values
                    comparison_df = pd.DataFrame({
                    'Actual': y_test.values.flatten(),
                    'Predicted': y_pred.flatten(),
                    })
                    st.dataframe(comparison_df)

                # Feature Importance
                feature_importances = model.feature_importances_
                importances = clf.named_steps['model'].feature_importances_
                if np.all(np.isnan(importances)):
                    st.error("Are you sure your data set is clean? Could there be a column that completely relates to the target column 1 to 1? Please re-evaluate data to ensure it does not have errors and is meaninful")
                else:
                    feature_importance_df = pd.DataFrame({'Feature': (numeric_cols+classification_cols), 'Importance': feature_importances})

                    # Sort the DataFrame by importance score in descending order
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                    st.subheader("Top 10 Feature Importances:")
                    st.dataframe(feature_importance_df)
            
                if regression_type == 'classification':
                    y_pred = label_encoder.inverse_transform(y_pred)

    
    except Exception as e:
        st.write(e)
        st.error(f"An error occurred: {e}")




