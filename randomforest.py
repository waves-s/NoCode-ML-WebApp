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

def get_feature_names(column_transformer):
    """Get feature names from column transformer"""
    output_features = []

    for name, pipe, features in column_transformer.transformers_:
        if name != "remainder":
            if hasattr(pipe, 'get_feature_names_out'):
                # For transformers with a get_feature_names_out method
                feature_names = pipe.get_feature_names_out(features)
            else:
                # For transformers without a get_feature_names_out method
                feature_names = features
            output_features.extend(feature_names)
    
    return output_features


def rmse_accuracy(regression_type, y_test, y_pred):
    if regression_type == 'numeric':
        rmse = root_mean_squared_error(y_test, y_pred)
        return rmse
        
    elif regression_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


def random_forest_optimizer(df, numeric_cols, classification_cols, target_column,regression_type):
    determine_feature_importance = False
    try:        
        best_rmse = float('inf')

        if regression_type == 'classification':
            best_rmse = float('-inf')
        
        # Feature Scaling - Use both StandardScaler and MinMaxScaler to determine best fit
        scalers = [('StandardScaler', StandardScaler()), ('MinMaxScaler', MinMaxScaler())]

        all_parameters = {}
        index = 0

        # Loop through each scaler
        for scaler_name, scaler in scalers:
           
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
                    y_pred, y_test, current_metric = model_creation(df, numeric_cols, classification_cols, target_column, regression_type, estimators, impurity, scaler, determine_feature_importance) 
                
                    if regression_type == 'classification':
                        if current_metric >= best_rmse:
                            best_rmse = current_metric
                    elif regression_type == 'numeric':
                        if current_metric <= best_rmse:
                            best_rmse = current_metric

                    
                    # Store RMSE/Accuracy and corresponding parameters
                    parameters = {'RMSE/Accuracy': current_metric, 'scaler': scaler, 'n_estimators': estimators, 'min_impurity': impurity}
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
        
        
def model_creation(df, numeric_cols, classification_cols, target_column, regression_type, estimators, impurity, scaler, determine_feature_importance):
    
    # Split the data into training and test sets
    X = df[numeric_cols + classification_cols]  
    y = df[target_column].values.ravel()
    
    # Define the model
    if regression_type == 'numeric':
        if impurity is not None:
            model = RandomForestRegressor(n_estimators=estimators, random_state=42, min_impurity_decrease=impurity)
        else:
            model = RandomForestRegressor(n_estimators=estimators, random_state=42)
                    
    elif regression_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)   
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[('scaler', scaler)])
    
    # Preprocessing for classification data
    # classification_transformer = Pipeline(steps=[('encoder', FunctionTransformer(label_encode_column, validate = False))])
    categories = [df[col].unique() for col in classification_cols]
    classification_transformer = OneHotEncoder(categories=categories, handle_unknown='ignore')
    # preprocess = Pipeline([('encoder', classification_transformer)])
    if classification_cols:
        classification_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
        # Bundle preprocessing for numeric and categorical data
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', classification_transformer, classification_cols)
        ])
    else:
        # Preprocessing for numeric data
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),])
        
    
    # Fit preprocessing pipeline
    preprocessor.fit(X_train)

    
    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', model)])
        
    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Get predictions
    y_pred = clf.predict(X_test)
    
    # Calculate RMSE/Accuracy
    current_metric = rmse_accuracy(regression_type, y_test, y_pred)

    if regression_type == 'classification':
        y_pred = label_encoder.inverse_transform(y_pred)
    
    if determine_feature_importance == True:
        # Get feature importances
        feature_names = get_feature_names(preprocessor) 
        importances = clf.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame(sorted(zip(importances, feature_names)), columns=['Value','Feature']).sort_values(by="Value", ascending=False,)
        feature_importance_df.columns = ['Importance', 'Feature']

        if np.all(np.isnan(importances)):
            st.error("Are you sure your data set is clean? Could there be a column that completely relates to the target column 1 to 1? Please re-evaluate data to ensure it does not have errors and is meaninful")
        else:
            st.subheader(":blue[Top 10 Feature Importances:]")
            st.dataframe(feature_importance_df.head(10))
            
    return y_pred, y_test, current_metric
    
      
def random_forest(df, numeric_cols, classification_cols, target_column, regression_type):

    try:
        #Run RandomForest with Default parameters
        scaler = StandardScaler()
        determine_feature_importance = False
        y_pred, y_test, current_metric = model_creation(df, numeric_cols, classification_cols, target_column, regression_type, 100, None,scaler, determine_feature_importance) 

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
            
            elif abs(current_metric-optimized_rsme_accuracy)/current_metric < 0.05:
                st.success(f"Best {wording} Achieved with default parameters, optimization does not add significant improvement.")      
            
            else:
                st.write(f":blue[Best Optimized {wording}: {optimized_rsme_accuracy}]")
                st.dataframe(df_params.sort_values(by='RMSE/Accuracy', ascending=True))

            determine_feature_importance = True
            y_pred, y_test, current_metric = model_creation(df, numeric_cols, classification_cols, target_column, regression_type,estimators, impurity, scaler_name, determine_feature_importance)

            # Calculate differences: positive values mean the prediction is higher than the actual
            if regression_type == 'numeric':
                differences = y_pred - y_test
                # .values.flatten()
                
                # Calculate the conservativeness metrics
                conservative_predictions = (differences > 0).sum()
                total_predictions = len(differences)
                percentage_conservative = (conservative_predictions / total_predictions) * 100
                st.write(f':blue[Percentage of Conservative Predictions (Prediction > Actual): {percentage_conservative:.2f}%]')

                # Identify non-conservative (lower than actual) predictions and their percentage
                non_conservative_predictions = (differences <= 0).sum()
                percentage_non_conservative = (non_conservative_predictions / total_predictions) * 100

                st.write(f':blue[Percentage of Non-Conservative Predictions (Prediction < Actual): {percentage_non_conservative:.2f}%]')

                # Create a DataFrame for comparing actual vs. predicted values
                comparison_df = pd.DataFrame({
                'Actual': y_test,
                # .values.flatten(),
                'Predicted': y_pred,
                # .flatten(),
                })
                st.subheader(":blue[Actual vs. Predicted Values for Target Column:]")
                st.dataframe(comparison_df)
            
            

    
    except Exception as e:
        st.write(e)
        st.error(f"An error occurred: {e}")




