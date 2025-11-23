import numpy as np

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

def get_best_features(df):
    
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
    th = 0.80

    Js_total = np.sum(sorted_Js)
    cum_sum = 0

    for J in sorted_Js:
        w = J / Js_total
        cum_sum += w
        
        if cum_sum > th:
            break

        ws.append(cum_sum)

    best_features = sorted_features[:len(ws)]

    print("Cumulative sum of ratios:", ws)
    print(f"Best {len(ws)} features are:", best_features)

    return best_features
