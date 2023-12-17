import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class HeadBrainSizePredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def display_dataset_info(self):
        print("Size of the actual dataset:", len(self.data))
        print("Columns of Dataset:", self.data.columns)

    def clean_prepare_data(self):
        # Extracting features and labels
        X = self.data['Head Size(cm^3)'].values.reshape(-1, 1)  # Feature: Head Size
        y = self.data['Brain Weight(grams)'].values  # Label: Brain Weight

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_linear_regression_model(self, X_train, y_train):
        # Create and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("R-squared Score:", r2)

    def visualize_model(self, model, X, y):
        # Visualize the linear regression line
        plt.scatter(X, y, color='blue')
        plt.plot(X, model.predict(X), color='red', linewidth=2)
        plt.title("Head Size vs. Brain Weight (Linear Regression)")
        plt.xlabel("Head Size (cm^3)")
        plt.ylabel("Brain Weight (grams)")
        plt.show()

def main():
    print("---- Head Brain Size Predictor by Nikhil Ahir ----")
    print("Linear Regression Application")
    print("Head Brain Size Predictor using Linear Regression")

    # Use forward slashes or double backslashes in the file path
    data_path = "/home/tesrekt/python/Assignment_16/headbrain.csv"

    brain_size_predictor = HeadBrainSizePredictor(data_path)

    brain_size_predictor.display_dataset_info()

    X_train, X_test, y_train, y_test = brain_size_predictor.clean_prepare_data()

    linear_regression_model = brain_size_predictor.train_linear_regression_model(X_train, y_train)

    brain_size_predictor.evaluate_model(linear_regression_model, X_test, y_test)

    brain_size_predictor.visualize_model(linear_regression_model, X_test, y_test)

if __name__ == "__main__":
    main()
