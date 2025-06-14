import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

PATCH_DIR = "stacked_patches_npy"
patch_files = [f for f in os.listdir(PATCH_DIR) if f.endswith(".npy")]

patch_ids = []
labels = []

for fname in patch_files:
    label_match = re.search(r'class(\d+)', fname)
    if label_match:
        label = int(label_match.group(1))
        if label == 1:
            continue  # Exclude class 1
        patch_ids.append(fname)
        labels.append(label)

print("Class distribution (after excluding class 1):", Counter(labels))

# Remove classes with less than 3 samples (to allow stratified 70/15/15 split)
class_counts = Counter(labels)
valid_classes = {cls for cls, count in class_counts.items() if count >= 3}
filtered = [(f, l) for f, l in zip(patch_ids, labels) if l in valid_classes]
patch_ids, labels = zip(*filtered)
patch_ids = list(patch_ids)
labels = list(labels)

print("Filtered class distribution:", Counter(labels))

# First split: 70% train, 30% temp
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(sss1.split(patch_ids, labels))
train_files = [patch_ids[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
temp_files = [patch_ids[i] for i in temp_idx]
temp_labels = [labels[i] for i in temp_idx]

# Second split: 15% val, 15% test from temp (split 0.5/0.5)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(temp_files, temp_labels))
val_files = [temp_files[i] for i in val_idx]
val_labels = [temp_labels[i] for i in val_idx]
test_files = [temp_files[i] for i in test_idx]
test_labels = [temp_labels[i] for i in test_idx]

# Save splits as CSV
pd.DataFrame({'filename': train_files, 'label': train_labels}).to_csv('train.csv', index=False)
pd.DataFrame({'filename': val_files, 'label': val_labels}).to_csv('val.csv', index=False)
pd.DataFrame({'filename': test_files, 'label': test_labels}).to_csv('test.csv', index=False)

print("Saved train.csv, val.csv, test.csv")