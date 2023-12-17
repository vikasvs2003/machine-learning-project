import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

class IrisSpeciesClassifier:
    def __init__(self):
        self.data = self.load_iris_data()

    def load_iris_data(self):
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        return data

    def split_data(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_decision_tree(self, X_train, X_test, y_train):
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X_train, y_train)
        return dt_classifier

    def evaluate_model(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def visualize_decision_tree(self, dt_classifier, feature_names, target_names):
        plt.figure(figsize=(12, 8))
        tree.plot_tree(dt_classifier, feature_names=feature_names, class_names=target_names, filled=True)
        plt.show()

def main():
    print("---- Marvellous Infosystem by Nikhil Ahir ----")
    print("-- Iris Species classification using Decision Tree algorithm --")

    iris_classifier = IrisSpeciesClassifier()
    X_train, X_test, y_train, y_test = iris_classifier.split_data()

    dt_classifier = iris_classifier.train_decision_tree(X_train, X_test, y_train)

    y_pred = dt_classifier.predict(X_test)

    iris_classifier.evaluate_model(y_test, y_pred)

    iris_classifier.visualize_decision_tree(dt_classifier, iris.feature_names, iris.target_names)

if __name__ == "__main__":
    main()
