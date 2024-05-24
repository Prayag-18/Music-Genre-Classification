import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def Load_Data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    X = np.array(data["mfcc"])  # X (ndarray): Inputs
    y = np.array(data["labels"])  # y (ndarray): Targets
    return X, y

if __name__ == "__main__":
    DATA_PATH = "data_10.json"
    X, y = Load_Data(DATA_PATH)

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    if hasattr(model.base_estimator_, 'feature_importances_'):
        feature_imps = np.mean([
            tree.feature_importances_ for tree in model.estimators_
        ], axis=0)
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(feature_imps)), feature_imps)
        plt.xlabel('Feature indices')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()
