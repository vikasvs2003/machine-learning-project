import pandas as pd
from seaborn import countplot
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class TitanicLogisticAnalysis:
    def __init__(self, dataset_path='TitanicDataset.csv'):
        self.titanic_data = pd.read_csv(dataset_path)

    def display_dataset_info(self):
        print("First 5 entries from loaded dataset:")
        print(self.titanic_data.head())
        print(f"Number of passengers: {len(self.titanic_data)}")

    def visualize_survival(self):
        print("Visualization: Survived or Non-Survived passengers")
        figure()
        target = "Survived"
        countplot(data=self.titanic_data, x=target).set_title(" Infosystem: Survived and Non-Survived passengers")
        show()

    def visualize_gender_survival(self):
        print("Visualization: Survived and Non-Survived passengers based on gender")
        figure()
        target = "Survived"
        countplot(data=self.titanic_data, x=target, hue="Sex").set_title(" Infosystem: Survived and Non-Survived passengers based on gender")
        show()

    def visualize_class_survival(self):
        print("Visualization: Survived and Non-Survived passengers based on passenger class")
        figure()
        target = "Survived"
        countplot(data=self.titanic_data, x=target, hue="Pclass").set_title(": Survived and Non-Survived passengers based on passenger class")
        show()

    def visualize_age_histogram(self):
        print("Visualization: Survived and Non-Survived passengers based on Age")
        figure()
        self.titanic_data["Age"].plot.hist().set_title(": Survived and Non-Survived passengers based on Age")
        show()

    def visualize_fare_histogram(self):
        print("Visualization: Survived and Non-Survived passengers based on Fare")
        figure()
        self.titanic_data["Fare"].plot.hist().set_title(": Survived and Non-Survived passengers based on Fare")
        show()

    def data_cleaning(self):
        print("Data cleaning: Removing 'zero' column")
        self.titanic_data.drop("zero", axis=1, inplace=True)
        print("First 5 entries from loaded dataset after removing 'zero' column:")
        print(self.titanic_data.head(5))

    def encode_categorical_data(self):
        print("Encoding categorical data:")
        print("Values of 'Sex' column:")
        print(pd.get_dummies(self.titanic_data["Sex"]))
        print("Values of 'Sex' column after removing one field:")
        sex = pd.get_dummies(self.titanic_data["Sex"], drop_first=True)
        print(sex.head(5))
        print("Values of 'Pclass' column after removing one field:")
        pclass = pd.get_dummies(self.titanic_data["Pclass"], drop_first=True)
        print(pclass.head(5))
        print("Values of data after concatenating new columns:")
        self.titanic_data = pd.concat([self.titanic_data, sex, pclass], axis=1)
        print(self.titanic_data.head(5))
        print("Values of data after removing irrelevant columns:")
        self.titanic_data.drop(5)

    def train_logistic_regression(self):
        print("Data training:")
        x = self.titanic_data.drop("Survived", axis=1)
        y = self.titanic_data["Survived"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        log_model = LogisticRegression()
        log_model.fit(x_train, y_train)

        print("Data testing:")
        prediction = log_model.predict(x_test)

        print("Calculate accuracy:")
        print("Classification report of logistic regression:")
        print(classification_report(y_test, prediction))
        print("Confusion Matrix of logistic regression:")
        print(confusion_matrix(y_test, prediction))
        print("Accuracy of logistic regression:")
        print(accuracy_score(y_test, prediction))


def main():
    print("--- Nikhil Ahir")
    print("Supervised machine learning")
    print("Logistic Regression on Titanic dataset")

    # Create an instance of TitanicLogisticAnalysis
    titanic_logistic_analysis = TitanicLogisticAnalysis()

    # Execute the analysis
    titanic_logistic_analysis.display_dataset_info()
    titanic_logistic_analysis.visualize_survival()
    titanic_logistic_analysis.visualize_gender_survival()
    titanic_logistic_analysis.visualize_class_survival()
    titanic_logistic_analysis.visualize_age_histogram()
    titanic_logistic_analysis.visualize_fare_histogram()
    titanic_logistic_analysis.data_cleaning()
    titanic_logistic_analysis.encode_categorical_data()
    titanic_logistic_analysis.train_logistic_regression()


if __name__ == "__main__":
    main()
