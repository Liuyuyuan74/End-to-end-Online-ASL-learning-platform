import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import os
import time
import joblib

# Constants
TEST_SIZE = .2
BASE_DIR = os.getcwd()
DATA_DIR = 'data'
DATA_FILE = 'data.pickle'
MODEL_DIR = 'models'
MODEL_FILE = 'aslModel.joblib'

# Paths
DATA_PATH = os.path.join(BASE_DIR, DATA_DIR, DATA_FILE)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR, MODEL_FILE)

def main():
    startTime = time.time()

    # Ensure the model directory exists
    if not os.path.exists(os.path.join(BASE_DIR, MODEL_DIR)):
        os.makedirs(os.path.join(BASE_DIR, MODEL_DIR))

    # Load the data
    with open(DATA_PATH, 'rb') as file:
        data_dict = pickle.load(file)

    data = np.array(data_dict['data'])
    labels = np.array(data_dict['labels'])

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, shuffle=True, stratify=labels)

    # Define the parameter grid

    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    }
    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Perform grid search
    print("Starting Grid Search...")
    grid_search.fit(x_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_predict = best_model.predict(x_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Optimized model accuracy: {accuracy*100:.2f}%")

    # Save the model
    joblib.dump({'model': best_model}, MODEL_PATH)

    # Execution time
    endTime = time.time()
    print(f"\nExecution Time: {(((endTime - startTime) * 10**3) / 1000)} Seconds")

if __name__ == "__main__":
    main()