import os

import numpy as np


DATA_PATH = os.path.join("RML2016.10b.dat", "CustomMOD-2026.a.h5")

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

    return None


def main():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Please install h5py before running this script.") from exc

    print("Inspecting dataset:", DATA_PATH)
    with h5py.File(DATA_PATH, "r") as f:
        print("Top-level keys:", list(f.keys()))

        x_key = find_dataset_key(f, ["X"])
        y_key = find_dataset_key(f, ["Y"])
        z_key = find_dataset_key(f, ["Z", "SNR", "snr"])

        if x_key is None or y_key is None or z_key is None:
            raise KeyError(f"Missing expected X/Y/Z datasets. Available keys: {list(f.keys())}")

        X = f[x_key]
        Y = f[y_key]
        Z = f[z_key]

        print("X shape:", X.shape, "dtype:", X.dtype)
        print("Y shape:", Y.shape, "dtype:", Y.dtype)
        print("Z shape:", Z.shape, "dtype:", Z.dtype)

        class_names = None
        if len(Y.shape) == 2:
            class_names = extract_class_names(f, y_key, Y.shape[1])
        elif len(Y.shape) == 1:
            class_names = extract_class_names(f, y_key, len(np.unique(Y[()])))

        if class_names is not None:
            print("Class names:", class_names)
        else:
            print("Class names: not found in file metadata")

        snr_values = np.unique(Z[()])
        print("Unique SNR count:", len(snr_values))
        print("SNR values:", snr_values.tolist())

        print(
            "SUMMARY | num_samples={} | x_shape={} | y_shape={} | z_shape={} | num_snrs={}".format(
                X.shape[0], X.shape, Y.shape, Z.shape, len(snr_values)
            )
        )


if __name__ == "__main__":
    main()
