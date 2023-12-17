import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

class PlayPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0)

    def display_dataset_info(self):
        print("Size of the actual dataset:", len(self.data))

    def clean_prepare_data(self):
        feature_names = ['Whether', 'Temperature']
        print("Names of features:", feature_names)

        # Extracting individual features
        weather = self.data['Whether']
        temperature = self.data['Temperature']
        play = self.data['Play']

        # Creating label encoder
        le = preprocessing.LabelEncoder()

        # Converting string labels into numbers
        weather_encoded = le.fit_transform(weather)
        print("Encoded Weather Labels:", weather_encoded)

        # Converting string labels into numbers
        temp_encoded = le.fit_transform(temperature)
        label = le.fit_transform(play)

        print("Encoded Temperature Labels:", temp_encoded)

        # Combining the weather and temperature into a single list of tuples
        features = list(zip(weather_encoded, temp_encoded))

        return features, label

    def train_knn_model(self, features, label):
        # Train the KNeighborsClassifier model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(features, label)
        return model

    def test_knn_model(self, model, new_data):
        # Test the model with new data
        predicted = model.predict([new_data])
        print("Predicted Play:", predicted)

def main():
    print("---- NIKhil ML by Nikhil Ahir ----")
    print("Machine Learning Application")
    print("Play Predictor Application using K Nearest Neighbor Algorithm")

    # Use forward slashes or double backslashes in the file path
    data_path = "/home/tesrekt/python/Assignment_16/PlayPredictor.csv"

    play_predictor = PlayPredictor(data_path)

    play_predictor.display_dataset_info()

    features, label = play_predictor.clean_prepare_data()

    knn_model = play_predictor.train_knn_model(features, label)

    # Example new data: Overcast and Mild
    new_data = [0, 2]

    play_predictor.test_knn_model(knn_model, new_data)

if __name__ == "__main__":
    main()
