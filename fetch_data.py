import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Setup
subjects = range(1, 11)
runs = [4, 8, 12]
custom_mapping = {'Rest': 1, 'Left_Fist': 2, 'Right_Fist': 3}

# We will store the final score for each person here
subject_scores = []

print(f"--- Training Personalized AI Models for {len(subjects)} Subjects ---\n")

for subject_id in subjects:
    try:
        # 1. Fetch and Load (Quietly)
        file_paths = mne.datasets.eegbci.load_data(subject_id, runs)
        raw_files = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in file_paths]
        raw = mne.concatenate_raws(raw_files)
        mne.datasets.eegbci.standardize(raw)
        
        # 2. Filter and Epoch (Quietly)
        raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', skip_by_annotation='edge', verbose=False)
        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=custom_mapping, tmin=-1.0, tmax=4.0, 
                            baseline=None, preload=True, verbose=False)
        
        # Isolate just this specific subject's Left vs Right data
        epochs_train = epochs['Left_Fist', 'Right_Fist']
        X = epochs_train.get_data()
        y = epochs_train.events[:, -1]
        
        # 3. Build and Train the Personalized AI
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        
        # 4. Test the Model
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        
        # To keep the terminal clean, we suppress the MNE logging inside the cross_val_score
        with mne.utils.use_log_level('ERROR'):
            scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
        
        mean_score = np.mean(scores) * 100
        subject_scores.append(mean_score)
        
        print(f"Subject {subject_id:2d} | Personalized Accuracy: {mean_score:.2f}%")
        
    except Exception as e:
        print(f"Subject {subject_id:2d} | Error: {e}")

# --- Final Results ---
print("\n--------------------------------------------------")
print(f"Universal Model Accuracy (Old Method): ~43.44%")
print(f"Personalized Models Average (New Method): {np.mean(subject_scores):.2f}%")
print("--------------------------------------------------")