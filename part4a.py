import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fisher_ratio(feature_names, df_1, df_2):

    Js = []
    
    for col_name in feature_names:

        mu_c1 = df_1[col_name].mean()
        std_c1 = df_1[col_name].std()

        mu_c2 = df_2[col_name].mean()
        std_c2 = df_2[col_name].std()

        J = (mu_c1 - mu_c2) ** 2 / (std_c1 ** 2 + std_c2 ** 2)

        Js.append(J)

    return np.array(Js)

def get_best_features(df, plot=False, threshold=0.8):
    
    mask_cold = df["Class"] == 1
    mask_excessive = df["Class"] == 2
    mask_good = df["Class"] == 3

    features = df.iloc[:, 2:]
    feature_names = features.columns

    df_cold = features[mask_cold]
    df_excessive = features[mask_excessive]
    df_good = features[mask_good]

    Js_good_cold = fisher_ratio(feature_names, df_good, df_cold)
    Js_good_excessive = fisher_ratio(feature_names, df_good, df_excessive)

    Js = Js_good_cold + Js_good_excessive

    sorted_indices = np.argsort(Js)

    sorted_features = feature_names[sorted_indices][::-1]
    sorted_Js = Js[sorted_indices][::-1]
    
    # Choose cumulative sum of ratios to be less than .80
    
    ws = []

    Js_total = np.sum(sorted_Js)
    cum_sum = 0

    for J in sorted_Js:
        w = J / Js_total
        cum_sum += w
        
        if cum_sum > threshold:
            break

        ws.append(cum_sum)

    best_features = sorted_features[:len(ws)]

    print("Cumulative sum of ratios:", ws)
    print(f"Best {len(ws)} features are:", best_features)

    if plot and len(ws) >= 3:
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_title(f"Separability of Best 3 Features")

        ax.set_xlabel(best_features[0])
        ax.set_ylabel(best_features[1])
        ax.set_zlabel(best_features[2])

        ax.grid(True)

        ax.scatter(
            df_cold[best_features[0]], df_cold[best_features[1]], df_cold[best_features[2]],
            color='blue', alpha=0.6, s=20, label="Cold Weld"
        )

        ax.scatter(
            df_excessive[best_features[0]], df_excessive[best_features[1]], df_excessive[best_features[2]],
            color='red', alpha=0.6, s=20, label="Excessive Weld"
        )

        ax.scatter(
            df_good[best_features[0]], df_good[best_features[1]], df_good[best_features[2]],
            color='green', alpha=0.6, s=20, label="Good Weld"
        )

        ax.legend()
        plt.show()

    return best_features

