import os
import tarfile
import pydicom
from collections import Counter

DATA_ROOT = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
SIDE = "right"

depths = []
patients = []

print("Scanning MRI volumes...\n")

for subset in sorted(os.listdir(DATA_ROOT)):
    subset_path = os.path.join(DATA_ROOT, subset)
    if not os.path.isdir(subset_path):
        continue

    for patient in sorted(os.listdir(subset_path)):
        patient_path = os.path.join(subset_path, patient)
        mri_path = os.path.join(patient_path, "mri", SIDE)
        if not os.path.isdir(mri_path):
            continue

        # Find the first .tar.gz file
        tar_files = [f for f in os.listdir(mri_path) if f.endswith(".tar.gz")]
        if not tar_files:
            continue

        tar_path = os.path.join(mri_path, tar_files[0])

        # Count DICOM slices inside tar
        try:
            temp_depth = 0
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    temp_depth += 1

            depths.append(temp_depth)
            patients.append(patient)
            if temp_depth != 161:
                print(f"{subset}/{patient}: depth = {temp_depth}")

        except Exception as e:
            print(f"Error reading {tar_path}: {e}")

# Summary
print("\n---------------------------")
print("SUMMARY OF MRI DEPTHS")
print("---------------------------")

if depths:
    print(f"Total MRIs: {len(depths)}")
    print(f"Min depth : {min(depths)}")
    print(f"Max depth : {max(depths)}")
    print(f"Unique depths: {sorted(set(depths))}")

    # Frequency count
    counts = Counter(depths)
    print("\nDepth frequency:")
    for d, c in sorted(counts.items()):
        print(f"  D = {d}: {c} patients")
else:
    print("No MRI volumes found!")
