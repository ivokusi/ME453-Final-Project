import numpy as np
import matplotlib.pyplot as plt

def error_rate(cm):
    return 1 - np.trace(cm) / np.sum(cm)

def class_errors(cm):
    return 1 - np.diag(cm) / cm.sum(axis=1)

def error_plots(cms, classifier_names):
    
    folds_ = np.arange(1,6,1)
    classes = ["Class 1", "Class 2", "Class 3"] 
    
    _, axs = plt.subplots(2,len(cms), sharey=True)
    
    for j, cm in enumerate(cms):
        error_rates = []
        print(classifier_names[j])

        for i in folds_:        
            print(f"Fold {i}")
            print(cm[i-1])
            print("#" * 20)
            
            error_rates.append(error_rate(cm[i-1]))
            ce = class_errors(cm[i-1])
            axs[0,j].plot(classes, ce, marker='o', label=f"Fold {i}")

        axs[0,j].set_title(f"Per-Class Error for {classifier_names[j]}")
        axs[0,j].set_ylabel("Error Rate")
        axs[0,j].set_ylim((0, 1))

        axs[0,j].legend()
        axs[0,j].grid(True)
        
        axs[1,j].bar(folds_, error_rates)
        axs[1,j].set_title(f"Overall Error Rate for {classifier_names[j]}")
        axs[1,j].set_ylabel("Error Rate")
        axs[1,j].set_xlabel("Fold number")
        axs[1,j].set_ylim((0, 1))
        axs[1,j].grid(True)
    
    # Add space between subplots to avoid label/title collisions
    plt.tight_layout()
        