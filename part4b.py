import numpy as np
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# We do not need to scale X as the data is already normalized

def classify(X, y, classifier, classifier_name, splits=5):
    
    kf = KFold(n_splits=splits, random_state=42, shuffle=True)

    accuracies = []
    confusion_matrices = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        X_train = X.iloc[train_index, :]
        X_test = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        ConfMatrix = confusion_matrix(y_test, y_pred)
        
        #print(f"Fold {i + 1}")
        #print(f"{classifier_name} Accuracy: {accuracy}")
        #print("#" * 70)
        
        accuracies.append(accuracy)
        confusion_matrices.append(ConfMatrix)
        
    avg_accuracy = np.mean(accuracies)
    print(f"Average {classifier_name} Accuracy: {avg_accuracy}")

    return accuracies, confusion_matrices
        