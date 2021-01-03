import sys
import numpy as np

accs = []
with open(sys.argv[1], "r") as fp:
    for line in fp:
        if "val_acc" in line.lower():
            accs.append(float(line.split()[-1]))
np.save("val_accs.npy", np.array(accs))
