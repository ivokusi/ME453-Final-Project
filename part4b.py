from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

# We do not need to scale X as the data is already normalized
def classify(X, y, classifier, splits=5, classifier_name_prefix=""):
    
    kf = KFold(n_splits=splits, random_state=42, shuffle=True)

    accuracies = []
    ConfMatrices = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        X_train = X.iloc[train_index, :]
        X_test = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        ConfMatrix = confusion_matrix(y_test, y_pred)
        
        print(f"Fold {i + 1}")
        print(f"{classifier_name_prefix}{classifier.__class__.__name__} Accuracy: {accuracy}")
        print("#" * 70)
        
        accuracies.append(accuracy)
        ConfMatrices.append(ConfMatrix)
        

    avg_accuracy = np.mean(accuracies)
    print(f"Average {classifier_name_prefix}{classifier.__class__.__name__} Accuracy: {avg_accuracy}")

    return ConfMatrices, accuracies
        