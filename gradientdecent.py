import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def gradeint_descent(df):

    # Prepare the feature matrix X and target vector y
    X = df[['Tool_Depth_Range_%WT', 'Length_mm']].values
    y = df['Field_Total_Depth_%_WT'].values
    st.write(X)

    # Normalize features for more efficient gradient descent
    X_normaliazed = (X - X.mean(axis=0)) / X.std(axis=0)
    st.write(X)
    # Add a column of ones to X to incorporate the bias term (intercept) in our model
    X_with_bias = np.hstack([np.ones((X_normaliazed.shape[0], 1)), X_normaliazed])
    st.write(X)
    # Initialize weights (including bias) with zeros
    theta = np.zeros(X_with_bias.shape[1])
    st.write(X_with_bias)
    st.write(theta)
    # Define the learning rate and number of iterations
    learning_rate = 0.01
    iterations = 1000

    # Gradient Descent Function
    def gradient_descent_iteration(X, y, theta, learning_rate, iterations):
        m = len(y)
        cost_history = np.zeros(iterations)
        # st.write(cost_history)
        
        for i in range(iterations):
            predictions = X.dot(theta)  # Predicts y by multiplying the vector X with theta
            errors = predictions - y  # Difference between predictions and actual y
            gradients = X.T.dot(errors) / m  # The gradient of the cost function = X transposed x Errors
            theta -= learning_rate * gradients  # Update the theta vector by taking a step proportional to the gradient
            cost_history[i] = (1/(2*m)) * np.sum(errors**2)  # Compute and record the cost
        st.write(theta)
        return theta, cost_history

    # Run Gradient Descent
    theta_final, cost_history = gradient_descent_iteration(X_with_bias, y, theta, learning_rate, iterations)

    # Plotting the cost history to visualize the reduction in cost over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Reduction over Iterations')
    st.pyplot(plt)  # Use st.pyplot() to render the matplotlib plot in Streamlit

    # Display the final parameters of the model
    st.write(f'Final Parameters (Theta): {theta_final}')
