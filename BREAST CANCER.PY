import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class BreastCancerClassifier:
    def __init__(self, input_path, output_path, headers):
        self.INPUT_PATH = input_path
        self.OUTPUT_PATH = output_path
        self.HEADERS = headers
        self.dataset = None

    def read_data(self):
        # Read the data into a pandas dataframe
        self.dataset = pd.read_csv(self.INPUT_PATH)

    def get_headers(self):
        return self.dataset.columns.values

    def add_headers(self):
        self.dataset.columns = self.HEADERS

    def data_file_to_csv(self):
        # Save the loaded dataset into csv format
        self.dataset.to_csv(self.OUTPUT_PATH, index=False)
        print("File saved ...!")

    def split_dataset(self, train_percentage, feature_headers, target_header):
        # Split dataset into train and test dataset
        train_x, test_x, train_y, test_y = train_test_split(self.dataset[feature_headers],
                                                            self.dataset[target_header],
                                                            train_size=train_percentage)
        return train_x, test_x, train_y, test_y

    def handle_missing_values(self, missing_values_header, missing_label):
        # Filter missing values from the dataset
        self.dataset = self.dataset[self.dataset[missing_values_header] != missing_label]

    def random_forest_classifier(self, features, target):
        # Train the random forest classifier with features and target data
        clf = RandomForestClassifier()
        clf.fit(features, target)
        return clf

    def dataset_statistics(self):
        # Display basic statistics of the dataset
        print(self.dataset.describe())

    def main(self):
        self.read_data()
        self.add_headers()
        self.data_file_to_csv()
        self.dataset_statistics()

        # Filter missing values
        self.handle_missing_values(self.HEADERS[6], '?')
        train_x, test_x, train_y, test_y = self.split_dataset(0.7, self.HEADERS[1:-1], self.HEADERS[-1])

        # Train and Test dataset size details
        print("Train_x Shape :: ", train_x.shape)
        print("Train_y Shape :: ", train_y.shape)
        print("Test_x Shape :: ", test_x.shape)
        print("Test_y Shape :: ", test_y.shape)

        # Create random forest classifier instance
        trained_model = self.random_forest_classifier(train_x, train_y)
        print("Trained model :: ", trained_model)

        predictions = trained_model.predict(test_x)
        for i in range(0, 205):
            print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

        print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
        print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
        print(" Confusion matrix ", confusion_matrix(test_y, predictions))

if __name__ == "__main__":
    # File Paths
    INPUT_PATH = "breast-cancer-wisconsin.data"
    OUTPUT_PATH = "breast-c
