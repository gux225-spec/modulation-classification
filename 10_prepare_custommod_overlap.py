import os

import numpy as np


DATA_PATH = os.path.join("RML2016.10b.dat", "CustomMOD-2026.a.h5")
OUT_PATH = os.path.join("data_cache", "custommod_overlap_eval.npz")

FALLBACK_CLASSES = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "16APSK",
    "32APSK",
    "2FSK",
    "4FSK",
    "MSK",
    "GMSK",
    "CPM",
    "OQPSK",
    "PI4DQPSK",
]


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def normalize_mod_name(name):
    text = decode_text(name).strip().upper()
    text = text.replace("\u03c0", "PI")
    text = text.replace("\u03a0", "PI")
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    text = text.replace(" ", "")
    return text


def map_custommod_to_model_label(name):
    alias_map = {
        "BPSK": "BPSK",
        "QPSK": "QPSK",
        "8PSK": "8PSK",
        "16QAM": "QAM16",
        "64QAM": "QAM64",
    }
    return alias_map.get(normalize_mod_name(name))


def maybe_class_name_list(values, num_classes):
    arr = np.asarray(values)
    if arr.ndim == 0 or len(arr) != num_classes:
        return None
    return [decode_text(v) for v in arr]


def find_dataset_key(h5_file, candidates):
    lower_map = {k.lower(): k for k in h5_file.keys()}
    for cand in candidates:
        key = lower_map.get(cand.lower())
        if key is not None:
            return key
    return None


def extract_class_names(h5_file, y_key, num_classes):
    attr_keys = ["classes", "class_names", "labels", "label_names", "mods", "modulations"]

    for holder in [h5_file[y_key], h5_file]:
        for attr_key in attr_keys:
            if attr_key in holder.attrs:
                names = maybe_class_name_list(holder.attrs[attr_key], num_classes)
                if names is not None:
                    return names

    for ds_key in attr_keys:
        if ds_key in h5_file:
            names = maybe_class_name_list(h5_file[ds_key][()], num_classes)
            if names is not None:
                return names

    if num_classes == len(FALLBACK_CLASSES):
        return list(FALLBACK_CLASSES)

    raise ValueError("Could not determine class names for CustomMOD dataset.")


def load_custommod_dataset(path):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Please install h5py before running this script.") from exc

    with h5py.File(path, "r") as f:
        x_key = find_dataset_key(f, ["X"])
        y_key = find_dataset_key(f, ["Y"])
        z_key = find_dataset_key(f, ["Z", "SNR", "snr"])

        if x_key is None or y_key is None or z_key is None:
            raise KeyError(f"Missing expected X/Y/Z datasets. Available keys: {list(f.keys())}")

        X = f[x_key][()]
        Y = f[y_key][()]
        Z = f[z_key][()]

        if Y.ndim == 2:
            class_names = extract_class_names(f, y_key, Y.shape[1])
            labels = np.array([class_names[i] for i in np.argmax(Y, axis=1)], dtype=object)
        elif Y.ndim == 1:
            if np.issubdtype(Y.dtype, np.integer):
                class_names = extract_class_names(f, y_key, len(np.unique(Y)))
                labels = np.array([class_names[int(i)] for i in Y], dtype=object)
            else:
                labels = np.array([decode_text(v) for v in Y], dtype=object)
        else:
            raise ValueError(f"Unsupported Y shape: {Y.shape}")

    return X, labels, np.asarray(Z).reshape(-1)


def convert_samples_to_model_shape(X):
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {X.shape}")
    if X.shape[1:] == (2, 128):
        return X.astype(np.float32, copy=False)
    if X.shape[1:] == (128, 2):
        return np.transpose(X, (0, 2, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported sample shape: {X.shape}")


def main():
    print("Loading CustomMOD dataset:", DATA_PATH)
    X_raw, y_raw, snr_raw = load_custommod_dataset(DATA_PATH)
    X_iq = convert_samples_to_model_shape(X_raw)

    mapped = np.array([map_custommod_to_model_label(label) for label in y_raw], dtype=object)
    keep_mask = mapped != None

    X_keep = X_iq[keep_mask]
    y_keep = mapped[keep_mask]
    snr_keep = snr_raw[keep_mask]

    print("Original samples:", len(y_raw))
    print("Kept overlap samples:", len(y_keep))
    print("Dropped unseen samples:", int(np.sum(~keep_mask)))

    print("Kept label counts:")
    kept_names, kept_counts = np.unique(y_keep, return_counts=True)
    for name, count in zip(kept_names, kept_counts):
        print("  {}: {}".format(name, int(count)))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X=X_keep,
        y=y_keep,
        snr=snr_keep,
    )

    print("Saved overlap evaluation set to:", OUT_PATH)
    print("X shape:", X_keep.shape)
    print("y shape:", y_keep.shape)
    print("snr shape:", snr_keep.shape)


if __name__ == "__main__":
    main()
