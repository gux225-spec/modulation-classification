import os
import os
import pickle
import numpy as np
from feature_extraction import extract_features
from feature_enhancer import extract_disambiguation_features
from sklearn.model_selection import train_test_split

# Avoid KMeans memory leak on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '1'
# 1) data path
DAT_PATH = r"E:\PycharmProject\DSCI441_MLProject_RML\RML2016.10b.dat\RML2016.10b.dat"

# 2) Output directory
OUT_DIR = "data_cache"
os.makedirs(OUT_DIR, exist_ok=True)

# 3) Sampling Strategy: Number of samples per (mod, snr)=2000
PER_KEY = 2000
SEED = 0

def load_rml_dat(path: str):
    with open(path, "rb") as f:
        Xd = pickle.load(f, encoding="latin1")
    return Xd

def build_feature_dataset(Xd: dict, per_key: int, seed: int):
    rng = np.random.default_rng(seed)
    keys = list(Xd.keys())
    mods = sorted({k[0] for k in keys})
    snrs = sorted({k[1] for k in keys})

    X_list, y_list, snr_list = [], [], []

    total_keys = len(keys)
    for idx_k, (mod, snr) in enumerate(keys, start=1):
        arr = np.asarray(Xd[(mod, snr)])  # (6000, 2, 128)
        n = arr.shape[0]
        take = min(per_key, n)
        chosen = rng.choice(n, size=take, replace=False)

        for i in chosen:
            # Extract base features
            base_features = extract_features(arr[i])
            # Extract new disambiguation features
            new_features = extract_disambiguation_features(arr[i]) # Pass the original sample
            # Concatenate features
            combined_features = np.concatenate([base_features, new_features])
            X_list.append(combined_features)
            y_list.append(mod)
            snr_list.append(snr)

        if idx_k % 20 == 0:
            print(f"processed keys: {idx_k}/{total_keys}  current samples: {len(X_list)}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    snr = np.array(snr_list, dtype=np.int16)

    meta = {
        "per_key": per_key,
        "seed": seed,
        "num_keys": total_keys,
        "mods": mods,
        "snrs": snrs,
        "feature_dim": X.shape[1],
        "num_samples": X.shape[0],
    }
    return X, y, snr, meta

def main():
    print("Loading data")
    Xd = load_rml_dat(DAT_PATH)
    print(f"Data loaded. Type: {type(Xd)}, Number of keys: {len(Xd.keys()) if isinstance(Xd, dict) else 'Not a dict'}")

    print("Building feature dataset...")
    X, y, snr, meta = build_feature_dataset(Xd, PER_KEY, SEED)

    print(f"Total samples: {len(X)}. Splitting into train/val/test with joint stratification...")

    # Create a joint stratification key using both modulation and SNR
    stratify_key = [f"{mod}_{s}" for mod, s in zip(y, snr)]

    # First split: 70% training, 30% temporary (for val/test)
    # Stratify by the joint key to ensure balanced classes in all sets
    X_train, X_temp, y_train, y_temp, snr_train, snr_temp, indices_train, indices_temp = train_test_split(
        X, y, snr, np.arange(len(y)),
        test_size=0.3,
        random_state=SEED,
        stratify=stratify_key
    )

    # Create a new stratification key for the temporary set based on original indices
    stratify_key_temp = [stratify_key[i] for i in indices_temp]

    # Second split: 50% of temporary set for validation, 50% for testing
    # This results in 15% validation and 15% test of the original dataset
    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp, y_temp, snr_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=stratify_key_temp
    )

    print("Splitting complete.")
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples:       {len(X_test)}")

    # --- Save split datasets ---
    print("Saving split datasets...")
    
    # We will save split files and no longer save the original large files
    # Define base path for easy naming
    base_name = f"perkey{PER_KEY}"

    # Save training set
    np.save(os.path.join(OUT_DIR, f"{base_name}_X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, f"{base_name}_y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, f"{base_name}_snr_train.npy"), snr_train)

    # Save validation set
    np.save(os.path.join(OUT_DIR, f"{base_name}_X_val.npy"), X_val)
    np.save(os.path.join(OUT_DIR, f"{base_name}_y_val.npy"), y_val)
    np.save(os.path.join(OUT_DIR, f"{base_name}_snr_val.npy"), snr_val)

    # Save test set
    np.save(os.path.join(OUT_DIR, f"{base_name}_X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, f"{base_name}_y_test.npy"), y_test)
    np.save(os.path.join(OUT_DIR, f"{base_name}_snr_test.npy"), snr_test)

    # Update and save meta file
    meta["split_info"] = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "stratified_by": "modulation_type (y)"
    }
    meta_path = os.path.join(OUT_DIR, f"meta_{base_name}.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved all split dataset files.")
    print("Meta file updated with split info:", meta_path)

if __name__ == "__main__":
    main()
