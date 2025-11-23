import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_best_features(df, plot_3_best=False):
    
    mask_cold = df["Class"] == 1
    mask_excessive = df["Class"] == 2
    mask_good = df["Class"] == 3

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

    best_features = feature_names[sorted_indices][::-1]
    ratios = Js[sorted_indices][::-1]
    
    ## Choose cumulative sum of ratios to be less than .80
    ws = []
    th = .80
    sum_ = np.sum(ratios)
    cum_sum = 0
    for r in ratios :
        w = r / sum_
        cum_sum += w
        
        if cum_sum > th:
            break
        ws.append(cum_sum)


    print("Cumulative sum of ratios : ", ws)
    print(f"Best {len(ws)} features are : " , feature_names[sorted_indices][::-1][:len(ws)])
    
    if plot_3_best:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_title("Separability of Best 3 Features")

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
        plt.subplots_adjust(right=0.85)

        plt.show()

    return feature_names[sorted_indices][::-1][:len(ws)]


def PCA_df(df, scree_plot_cum, plot_PCs):
    X = df.iloc[:, 2:]  # Features only

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    pca = PCA()
    pca.fit_transform(X_scaled)

    # E. Variance Ratio 
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()

    # Scree Plot and Cum.
    if scree_plot_cum :
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Scree plot (E. Variance by Component)
        ax[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7)
        ax[0].set_title('Scree Plot - Explained Variance per Principal Component')
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Variance Ratio')
        ax[0].set_xticks(range(1, len(explained_variance_ratio) + 1))
        ax[0].grid(True)

        # Cumulative Explained Variance Plot
        ax[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        ax[1].set_title('Cumulative Explained Variance')
        ax[1].set_xlabel('Number of Components')
        ax[1].set_ylabel('Cumulative Explained Variance')
        ax[1].grid(True)


        plt.tight_layout()  
        plt.show()
    
    # lindos colores
    colors = ['#e41a1c', '#377eb8', '#4daf4a']#, '#984ea3', '#ff7f00',
          #'#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']

    color_map = {category: colors[i % len(colors)] for i, category in enumerate(df['Class'].unique())}
    color_values = df['Class'].map(color_map)

    pca_ = PCA(n_components=3)
    X_pca = pca_.fit_transform(X_scaled)

    if plot_PCs :
        # Step 3: Create a 3D scatter plot for the first three principal components
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for the first 3 principal components
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=color_values, s=100, marker='o')

        # Add titles and labels
        ax.set_title('PCA - First 3 Principal Components')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        # Add custom legend for categories
        for category, color in color_map.items():
            ax.scatter([], [], color=color, label=f'Quality {category}', s=100)  # Invisible points for legend
        ax.legend(title="Quality of Welds across 3 PCs")

        # Show the plot
        plt.show()


data4 = "./data/part4.csv"
df_data4 = pd.read_csv(data4)
#PCA_df(df_data4, False, True)
plot_best_features(df_data4)
