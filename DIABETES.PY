import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class DiabetesPredictor:
    def __init__(self, data_path):
        self.diabetes = pd.read_csv(data_path)

    def display_information(self):
        print("---- Nikhil ML by Nikhil Ahir ---")
        print("-- Diabetes predictor using the Decision Tree ---")

        print("Columns of Dataset:")
        print(self.diabetes.columns)

        print("First 5 records of dataset:")
        print(self.diabetes.head())

        print("Dimension of diabetes Data: {}".format(self.diabetes.shape))

    def train_decision_tree(self, max_depth=3, random_state=0):
        X_train, X_test, y_train, y_test = train_test_split(self.diabetes.loc[:, self.diabetes.columns != 'Outcome'],
                                                            self.diabetes['Outcome'], stratify=self.diabetes['Outcome'],
                                                            random_state=66)

        tree = DecisionTreeClassifier(random_state=random_state)
        tree.fit(X_train, y_train)

        print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        tree.fit(X_train, y_train)

        print("Accuracy on training set with max_depth={}: {:.3f}".format(max_depth, tree.score(X_train, y_train)))
        print("Accuracy on test set with max_depth={}: {:.3f}".format(max_depth, tree.score(X_test, y_test)))

        print("Feature importances:\n{}".format(tree.feature_importances_))

        self.plot_feature_importances_diabetes(tree)

    @staticmethod
    def plot_feature_importances_diabetes(model):
        plt.figure(figsize=(8, 6))
        n_features = 8
        plt.barh(range(n_features), model.feature_importances_, align='center')
        diabetes_features = [x for i, x in enumerate(model.columns) if i != 8]
        plt.yticks(np.arange(n_features), diabetes_features)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        plt.show()


def main():
    diabetes_predictor = DiabetesPredictor('C:/Users/NIKHIL/Desktop/python programs/machine learning/diabetes.csv')
    diabetes_predictor.display_information()
    diabetes_predictor.train_decision_tree(max_depth=3, random_state=0)


if __name__ == "__main__":
    main()
