import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_class_feature_histograms(df, selected_features):

    mask_cold = df["quality label"] == "I"
    mask_excessive = df["quality label"] == "II"
    mask_good = df["quality label"] == "III"

    features = df.iloc[:, 2:]

    df_cold = features[mask_cold]
    df_excessive = features[mask_excessive]
    df_good = features[mask_good]

    dfs = [df_cold, df_excessive, df_good]
    labels = ["Cold", "Excessive", "Good"]

    # For each feature, show its histogram for all three classes, in a row side by side
    
    for feature_name in selected_features:

        feature_range = (features[feature_name].min(), features[feature_name].max())
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        for i, (df_class, label) in enumerate(zip(dfs, labels)):
            
            ax[i].hist(df_class[feature_name], range=feature_range, bins=15, alpha=0.7)
            ax[i].set_title(label)

            ax[i].tick_params(axis='x', rotation=45)

        fig.suptitle(feature_name)

    plt.tight_layout()
    plt.show()

def plot_shewart_control_chart(df, selected_features, k=3):

    mask_cold = df["quality label"] == "I"
    mask_excessive = df["quality label"] == "II"
    mask_good = df["quality label"] == "III" 

    features = df.iloc[:, 2:]

    df_cold = features[mask_cold]
    df_excessive = features[mask_excessive]
    df_good = features[mask_good]

    control_limits = [] # (LCL, UCL) tuple

    for feature in selected_features:

        mu = df_good[feature].mean()
        sigma = df_good[feature].std()

        UCL = mu + k * sigma
        CL = mu
        LCL = mu - k * sigma

        plt.figure(figsize=(10,8))
        
        plt.scatter(df_good.index, df_good[feature], color="blue", label="Good")
        plt.scatter(df_excessive.index, df_excessive[feature], color="red", label="Excessive (non-conforming)")
        plt.scatter(df_cold.index, df_cold[feature], color="orange", label="Cold (non-conforming)")

        plt.hlines(UCL, xmin=df.index.min(), xmax=df.index.max(), colors="red", linestyles="dashed", label="UCL")
        plt.hlines(CL, xmin=df.index.min(), xmax=df.index.max(), colors="blue", linestyles="solid", label="CL")
        plt.hlines(LCL, xmin=df.index.min(), xmax=df.index.max(), colors="red", linestyles="dashed", label="LCL")

        plt.title(f"{feature} Shewhart Control Chart (k=2) Classification")
        plt.ylabel(feature)
        plt.xlabel("Experiment Number")

        plt.legend()

        plt.tight_layout()
        plt.show()

        control_limits.append((LCL, UCL))

    return control_limits

def get_misclassification_rates(df, selected_features, control_limits):

    misclassification_rates = []

    for selected_feature, (LCL, UCL) in zip(selected_features, control_limits):

        feature = pd.DataFrame()

        feature["expected"] = np.where(df["quality label"] == "III", "good", "bad")
        feature["prediction"] = np.where((LCL <= df[selected_feature]) & (df[selected_feature] <= UCL), "good", "bad")
        feature["classification"] = np.where(feature["expected"] == feature["prediction"], 1, 0) # Classified Correctly: 1   Misclassified: 0

        accuracy = sum(feature["classification"]) / feature.shape[0]
        misclassification_rate = 1 - accuracy

        misclassification_rates.append(misclassification_rate)

    return misclassification_rates
