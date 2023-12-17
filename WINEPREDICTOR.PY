import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

class WinePredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.feature_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'Proline']

    def display_dataset_info(self):
        print("Size of the actual dataset:", len(self.data))
        print("Names of features:", self.feature_names)

    def encode_labels(self):
        le = preprocessing.LabelEncoder()
        labels_encoded = le.fit_transform(self.data['Class'])
        print("Encoded Labels:", labels_encoded)

    def encode_features(self):
        for feature_name in self.feature_names:
            feature = self.data[feature_name]
            le = preprocessing.LabelEncoder()
            feature_encoded = le.fit_transform(feature)
            print(f"{feature_name}_encoded:", feature_encoded)

    def train_knn_model(self):
        # Create a list of encoded features
        encoded_features = [self.data[feature_name + '_encoded'] for feature_name in self.feature_names]

        # Combine encoded features into a single list
        features_combined = list(zip(*encoded_features))

        # Initialize and train KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=12)
        model.fit(features_combined, self.data['Class_encoded'])

        # Make predictions
        predicted = model.predict([[1, 3]])  # Example input, you can change this
        print("Predicted Class:", predicted)

def main():
    print("---- NIkhil ML by NIkhil Ahir ----")
    print("Machine learning application")
    print("Wine predictor application using k-nearest neighbor algorithm")

    wine_predictor = WinePredictor("C:/Users/NIKHIL/Desktop/python programs/machine learning/WinePredictor.csv")

    wine_predictor.display_dataset_info()
    wine_predictor.encode_labels()
    wine_predictor.encode_features()
    wine_predictor.train_knn_model()

if __name__ == "__main__":
    main()
