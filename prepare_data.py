"""One-time data preparation. Downloads the two UCI datasets used by the
real-data experiments and writes concrete.csv / energy.csv next to this script.
After downloading, the script validates the files against known SHA-256 checksums
to confirm that the data matches the copies used to produce the reported results.

Run:
    pip install ucimlrepo pandas openpyxl
    python prepare_data.py
"""
import hashlib
import os
import sys
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    sys.exit("Please install ucimlrepo:  pip install ucimlrepo")

OUT = os.path.dirname(os.path.abspath(__file__))

# SHA-256 checksums of the CSV files used in the reported experiments.
# If these do not match after downloading, the UCI repository may have been
# updated; re-run the experiment scripts and regenerate the result JSON files.
CHECKSUMS = {
    "concrete.csv": "d7d8bd087f832935e902bcb2687667238cac3fe06799677a93ca2f46dce8db02",
    "energy.csv":   "46fc2a38d43879311f0bd09441dbcb02594bc41f11a382c452f095fa1d752c18",
}


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --- Concrete Compressive Strength (UCI id 165, Yeh 1998) ---
print("Downloading Concrete Compressive Strength (UCI id 165)...")
c = fetch_ucirepo(id=165)
dfc = pd.concat([c.data.features, c.data.targets], axis=1)
dfc.columns = ["cement", "slag", "ash", "water", "superplastic",
               "coarseagg", "fineagg", "age", "strength"]
concrete_path = os.path.join(OUT, "concrete.csv")
dfc.to_csv(concrete_path, index=False)
print(f"  concrete.csv  {dfc.shape}  rows x cols")

# --- Energy Efficiency ENB2012 (UCI id 242, Tsanas & Xifara 2012) ---
print("Downloading Energy Efficiency (UCI id 242)...")
e = fetch_ucirepo(id=242)
dfe = pd.concat([e.data.features, e.data.targets], axis=1)
dfe.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
energy_path = os.path.join(OUT, "energy.csv")
dfe.to_csv(energy_path, index=False)
print(f"  energy.csv    {dfe.shape}  rows x cols")

# --- Checksum validation ---
print("\nValidating checksums...")
all_ok = True
for fname, expected in CHECKSUMS.items():
    path = os.path.join(OUT, fname)
    actual = sha256_file(path)
    match = actual == expected
    status = "OK" if match else "MISMATCH"
    print(f"  {status}  {fname}")
    if not match:
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
        print(f"    The UCI repository may have been updated. Re-run the experiment")
        print(f"    scripts and regenerate the result JSON files before submission.")
        all_ok = False

if all_ok:
    print("All checksums match. Data files are identical to the reported results.")
else:
    sys.exit(1)
