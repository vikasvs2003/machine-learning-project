import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class IrisSpeciesClassifier:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def display_dataset_info(self):
        print("Size of the actual dataset:", len(self.data))
        print("Columns of Dataset:", self.data.columns)

    def clean_prepare_data(self):
        # Extracting features and labels
        X = self.data.drop('Species', axis=1)  # Features
        y = self.data['Species']  # Labels

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_knn_model(self, X_train, y_train, n_neighbors=3):
        # Create and train the K Nearest Neighbors (KNN) model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)

        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", confusion)

    def main(self):
        print("---- Iris Species Classifier by Piyush Khairnar ----")
        print("K Nearest Neighbors (KNN) Algorithm")
        print("Iris Species Classification using KNN")

        self.display_dataset_info()

        X_train, X_test, y_train, y_test = self.clean_prepare_data()

        # Choose the value of k (number of neighbors) based on your requirement
        k_neighbors = 3
        knn_model = self.train_knn_model(X_train, y_train, n_neighbors=k_neighbors)

        self.evaluate_model(knn_model, X_test, y_test)

if __name__ == "__main__":
    data_path = "/home/tesrekt/python/Assignment_16/iris.csv"
    iris_classifier = IrisSpeciesClassifier(data_path)
    iris_classifier.main()
