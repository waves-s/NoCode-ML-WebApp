#Neural Network models can model complex non-linear relationships
# Can artifically add x% to generate more conservative results y_pred = y_pred + 0.05
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras import layers, models
import numpy as np


def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # Output layer for regression; no activation function
    return model

def custom_loss(y_true, y_pred):
    # Define custom loss function here if needed
    return tf.reduce_mean(tf.square(y_true - y_pred))

def tensor_flow_train_model(df, numeric_cols, categorical_cols, target_column):
    # Preprocess data
    X = df[numeric_cols + categorical_cols]
    y = df[target_column].astype('float32')
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(X_test)
    
    # This is a placeholder, adjust according to your actual data structure
    tool_depth_values = X_test['Tool_Depth_Range_%WT'].values
    
    # Preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        sparse_threshold=0.0
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # X_train = X_train.toarray()
    # X_test = X_test.toarray()
    
    # Convert to TensorFlow tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    # Build and compile the model
    model = build_model(X_train.shape[1])
    model.compile(optimizer='adam', loss = 'mean_squared_error')  # Use 'mean_squared_error' if custom_loss is not needed
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = .001), loss = 'mean_squared_error')
    # model.compile(optimizer='adam', loss=custom_loss)  # Use 'mean_squared_error' if custom_loss is not needed
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
    
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    st.subheader("TensorFlow Results:")
    st.write(f'Test Loss: {test_loss}')
    
    # Plot training history
    # st.line_chart(history.history['loss'])
    # st.line_chart(history.history['val_loss'])
    
    # Make predictions on the test set
    y_pred = model.predict(X_test).flatten()  # Flatten in case y_pred shape is (n, 1) instead of (n,)

    # Calculate RMSE
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_test))).numpy()
    st.write(f'Root Mean Squared Error: {rmse}')

    # Generate DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Tool_Depth_%WT': tool_depth_values
    })
    st.dataframe(comparison_df)

    # Calculate differences: positive values mean the prediction is higher than the actual (conservative)
    differences = y_pred - y_test

    # Calculate the conservativeness metrics
    conservative_predictions = np.sum(differences >= 0)
    total_predictions = len(differences)
    percentage_conservative = (conservative_predictions / total_predictions) * 100

    st.write(f'Percentage of Conservative Predictions: {percentage_conservative:.2f}%')

    # Identify non-conservative (lower than actual) predictions and their percentage
    non_conservative_predictions = np.sum(differences < 0)
    percentage_non_conservative = (non_conservative_predictions / total_predictions) * 100

    st.write(f'Percentage of Non-Conservative Predictions: {percentage_non_conservative:.2f}%')


