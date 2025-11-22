import matplotlib.pyplot as plt
import numpy as np

def plot_best_features(df):
    
    mask_cold = df["quality label"] == "I"
    mask_excessive = df["quality label"] == "II"
    mask_good = df["quality label"] == "III"

    features = df.iloc[:, 2:]
    feature_names = features.columns

    df_cold = features[mask_cold]
    df_excessive = features[mask_excessive]
    df_good = features[mask_good]

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

    Js_good_cold = fisher_ratio(feature_names, df_good, df_cold)
    Js_good_excessive = fisher_ratio(feature_names, df_good, df_excessive)

    Js = Js_good_cold + Js_good_excessive

    sorted_indices = np.argsort(Js)

    best_features = feature_names[sorted_indices][::-1][:3]
    _ = Js[sorted_indices][::-1][:3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Separability of Best 3 Features")

    ax.set_xlabel("Rise Peak")
    ax.set_ylabel("Rise Slope")
    ax.set_zlabel("Welding Pressure")

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

