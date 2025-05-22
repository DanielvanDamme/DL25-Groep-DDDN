from sklearn.preprocessing import Normalizer
import numpy as np

def normalize(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array) -> tuple[np.array, np.array, np.array, np.array, Normalizer, Normalizer]:
    
    X_normalizer = Normalizer()
    Y_normalizer = Normalizer()

    # Fit the normalizer to the data
    X_normalizer.fit(X_train)
    Y_normalizer.fit(y_train)

    # Transform the data
    X_train_normalized = X_normalizer.transform(X_train)
    X_test_normalized = X_normalizer.transform(X_test)
    y_train_normalized = Y_normalizer.transform(y_train)
    y_test_normalized = Y_normalizer.transform(y_test)

    return X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, X_normalizer, Y_normalizer

if __name__ == "__main__": # Als dit het hoofdprogramma is, voer deze uit
    # Example usage
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[10, 11, 12], [13, 14, 15]])
    y_test = np.array([[7, 8], [9, 10]])
    X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, X_normalizer, Y_normalizer = normalize(X_train, y_train, X_test, y_test)
    print("X_train_normalized:", X_train_normalized)
    print("y_train_normalized:", y_train_normalized)
    print("X_test_normalized:", X_test_normalized)
    print("y_test_normalized:", y_test_normalized)