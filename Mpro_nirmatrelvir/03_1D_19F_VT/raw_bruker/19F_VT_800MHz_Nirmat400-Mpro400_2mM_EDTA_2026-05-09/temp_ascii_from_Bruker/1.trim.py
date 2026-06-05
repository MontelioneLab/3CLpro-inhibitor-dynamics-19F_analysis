import numpy as np
import glob

TARGET_SIZE = 16578 

for fname in glob.glob("asci_*.txt"):
    # read file, skip comment lines
    with open(fname) as f:
        lines = [l for l in f if not l.startswith("#")]

    data = np.array([float(l.strip()) for l in lines])

    if len(data) == TARGET_SIZE + 1:
        data = data[:TARGET_SIZE]   # drop last point
        print(f"{fname}: trimmed 1 point")
    elif len(data) == TARGET_SIZE:
        print(f"{fname}: OK")
    else:
        print(f"{fname}: unexpected size {len(data)}")

    np.savetxt(fname.replace(".txt", "_trim.txt"), data)

