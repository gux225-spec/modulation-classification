import os
import pickle
import numpy as np
from collections import Counter

from feature_extraction import extract_features

MINI_PATH = os.path.join("data_cache", "rml2016_10b_mini.pkl")

def main():
    with open(MINI_PATH, "rb") as f:
        Xd_mini = pickle.load(f)

    keys = list(Xd_mini.keys())
    print("mini type:", type(Xd_mini), "num keys:", len(keys))
    print("example key:", keys[0])

    # Take some samples and run feature extraction
    X_feat = []
    y = []
    snr = []

    # Select up to 5 records from each key for testing
    for (mod, s), arr in Xd_mini.items():
        arr = np.asarray(arr)
        take = min(5, arr.shape[0])
        for i in range(take):
            fvec = extract_features(arr[i])
            X_feat.append(fvec)
            y.append(mod)
            snr.append(s)

    X_feat = np.vstack(X_feat)
    y = np.array(y)
    snr = np.array(snr)

    print("feature matrix shape:", X_feat.shape)  # (N, D)
    print("feature dim D =", X_feat.shape[1])
    print("any NaN?", np.isnan(X_feat).any(), "any Inf?", np.isinf(X_feat).any())
    print("label counts (top 5):", Counter(y).most_common(5))
    print("snr unique (first 10):", np.unique(snr)[:10])

    # Print the first eigenvector as a sanity check
    print("first feature vector (first 10 dims):", X_feat[0][:10])

if __name__ == "__main__":
    main()
