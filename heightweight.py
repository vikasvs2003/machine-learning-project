import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class HeightWeightPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def display_dataset_info(self):
        print("Size of the actual dataset:", len(self.data))
        print("Columns of Dataset:", self.data.columns)

    def clean_prepare_data(self):
        # Extracting features and labels
        X = self.data['Height'].values.reshape(-1, 1)  # Feature: Height
        y = self.data['Weight'].values  # Label: Weight

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_regression_model(self, X_train, y_train):
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
        plt.title("Height vs. Weight (Linear Regression)")
        plt.xlabel("Height (cm)")
        plt.ylabel("Weight (kg)")
        plt.show()

def main():
    print("---- Height Weight Predictor by NIkhil Ahir ----")
    print("Linear Regression Application")
    print("Height Weight Predictor using Linear Regression")

    # Use forward slashes or double backslashes in the file path
    data_path = "/home/tesrekt/python/Assignment_16/heightweight.csv"

    height_weight_predictor = HeightWeightPredictor(data_path)

    height_weight_predictor.display_dataset_info()

    X_train, X_test, y_train, y_test = height_weight_predictor.clean_prepare_data()

    regression_model = height_weight_predictor.train_regression_model(X_train, y_train)

    height_weight_predictor.evaluate_model(regression_model, X_test, y_test)

    height_weight_predictor.visualize_model(regression_model, X_test, y_test)

if __name__ == "__main__":
    main()
