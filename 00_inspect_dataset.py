import pickle
import numpy as np

path = r"E:\PycharmProject\DSCI441_MLProject_RML\RML2016.10b.dat\RML2016.10b.dat"

with open(path, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

print("type(Xd):", type(Xd))

# dict，key=(mod, snr)，value=array
if isinstance(Xd, dict):
    keys = list(Xd.keys())
    print("num keys:", len(keys))
    print("example key:", keys[0])

    v = Xd[keys[0]]
    v = np.array(v)
    print("value type:", type(v), "shape:", v.shape, "dtype:", v.dtype)

    mods = sorted({k[0] for k in keys})
    snrs = sorted({k[1] for k in keys})
    print("num mods:", len(mods))
    print("mods:", mods)
    print("num snrs:", len(snrs))
    print("snrs (first 10):", snrs[:10], "... last 5:", snrs[-5:])

    # verify the I/Q dimension order: typically (N,2,128) or (N,128,2)
    if v.ndim == 3:
        print("One sample shape:", v[0].shape)
        print("first few numbers:", v[0].ravel()[:10])
else:
    # Rare case: Provide general information when it is not a dictionary
    arr = np.array(Xd)
    print("as array shape:", arr.shape, "dtype:", arr.dtype)
