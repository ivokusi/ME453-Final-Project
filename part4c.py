import numpy as np
import matplotlib.pyplot as plt

def error_rate(cm):
    return 1 - np.trace(cm) / np.sum(cm)

def class_errors(cm):
    return 1 - np.diag(cm) / cm.sum(axis=1)


def error_plots(Cmatrix, classifier_name, type_kernel=''):
    folds_ = np.arange(1,6,1)
    classes = ["Class 1", "Class 2", "Class 3"] 
    error_rates = []

    plt.figure(figsize=(7,5))
    for i in folds_:
        print(f"Fold {i}")
        print(Cmatrix[i-1])
        print("#"*20)
        error_rates.append(error_rate(Cmatrix[i-1]))
        ce = class_errors(Cmatrix[i-1])
        plt.plot(classes, ce, marker='o', label=f"Fold {i}")

    plt.title(f"Per-Class Error for all Folds for {classifier_name} classifier {type_kernel}")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.bar(folds_, error_rates)
    plt.title(f"Overall Error Rate by Folds for {classifier_name} classifier {type_kernel}")
    plt.ylabel("Error Rate")
    plt.grid(True)
    plt.show()