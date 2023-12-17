import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class SalesPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def prepare_data(self):
        X = self.data[['TV', 'Radio', 'Newspaper']]
        y = self.data['Sales']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_linear_regression_model(self, X_train, X_test, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def predict_and_evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print("R-squared Score:", r2)

def main():
    print("---- VIKAS BHONDE ML ---")
    print("Machine Learning Application")
    print("Sales Predictor using Linear Regression")

    # Use forward slashes or double backslashes in the file path
    data_path = "/home/tesrekt/python/Assignment_16/advertising.csv"
    
    sales_predictor = SalesPredictor(data_path)
    X_train, X_test, y_train, y_test = sales_predictor.prepare_data()

    model = sales_predictor.train_linear_regression_model(X_train, X_test, y_train)
    
    sales_predictor.predict_and_evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
