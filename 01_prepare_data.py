import os
import pickle
import numpy as np

DAT_PATH = r"E:\PycharmProject\DSCI441_MLProject_RML\RML2016.10b.dat\RML2016.10b.dat"

OUT_DIR = "data_cache"
os.makedirs(OUT_DIR, exist_ok=True)

def load_rml_dat(path: str):
    with open(path, "rb") as f:
        Xd = pickle.load(f, encoding="latin1")
    return Xd

def inspect_rml_dict(Xd: dict):
    keys = list(Xd.keys())
    mods = sorted({k[0] for k in keys})
    snrs = sorted({k[1] for k in keys})

    v0 = np.array(Xd[keys[0]])
    print("type(Xd):", type(Xd))
    print("num keys:", len(keys))
    print("example key:", keys[0])
    print("value shape:", v0.shape, "dtype:", v0.dtype)
    print("num mods:", len(mods))
    print("mods:", mods)
    print("num snrs:", len(snrs))
    print("snrs head:", snrs[:10], "tail:", snrs[-5:])
    return mods, snrs

def save_mini_sample(Xd: dict, out_path: str, per_key: int = 20, seed: int = 0):
    """
    从每个 (mod, snr) key 抽取 per_key 条样本，保存成一个更小的 dict
    用于后续快速调试（特征、模型等）
    """
    rng = np.random.default_rng(seed)
    mini = {}
    for k, arr in Xd.items():
        arr = np.array(arr)
        n = arr.shape[0]
        idx = rng.choice(n, size=min(per_key, n), replace=False)
        mini[k] = arr[idx]
    with open(out_path, "wb") as f:
        pickle.dump(mini, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    Xd = load_rml_dat(DAT_PATH)
    mods, snrs = inspect_rml_dict(Xd)

    mini_path = os.path.join(OUT_DIR, "rml2016_10b_mini.pkl")
    save_mini_sample(Xd, mini_path, per_key=20, seed=0)
    print(f"Saved mini sample to: {mini_path}")
