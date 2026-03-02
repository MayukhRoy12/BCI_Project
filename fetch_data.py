import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

# 1. Fetch and load data
subject_id = 1
runs = [4, 8, 12]
file_paths = mne.datasets.eegbci.load_data(subject_id, runs)
raw_files = [mne.io.read_raw_edf(f, preload=True) for f in file_paths]
raw = mne.concatenate_raws(raw_files)
mne.datasets.eegbci.standardize(raw)

# 2. Filter the data (8-30 Hz Mu/Beta rhythms)
raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', skip_by_annotation='edge')

# 3. Extract Events & Create Epochs
events, event_dict = mne.events_from_annotations(raw)
custom_mapping = {'Rest': 1, 'Left_Fist': 2, 'Right_Fist': 3}
epochs = mne.Epochs(raw, events, event_id=custom_mapping, tmin=-1.0, tmax=4.0, 
                    baseline=None, preload=True)

# --- NEW CODE: MACHINE LEARNING ---
print("\n--- Training the AI ---")

# Isolate just the Left and Right Fist events for binary classification
epochs_train = epochs['Left_Fist', 'Right_Fist']

# Extract the data matrix (X) and the target labels (y)
X = epochs_train.get_data() # The brainwaves
y = epochs_train.events[:, -1] # The answers (2 for Left, 3 for Right)

# Build the AI Pipeline
# 1. CSP extracts the best spatial features (reducing 64 channels to 4 core patterns)
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
# 2. LDA classifies those patterns
lda = LinearDiscriminantAnalysis()

# Assemble them into a single pipeline
clf = Pipeline([('CSP', csp), ('LDA', lda)])

# Cross-Validation: Train and test the model 10 times on different random chunks of data
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

# Print the final results
print(f"\nAI Classification Accuracy:")
print(f"Mean: {np.mean(scores)*100:.2f}%")
print(f"Scores across 10 tests: {[round(s*100, 1) for s in scores]}")