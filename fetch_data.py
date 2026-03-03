import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score


subjects = range(1, 11)
runs = [4, 8, 12]
custom_mapping = {'Rest': 1, 'Left_Fist': 2, 'Right_Fist': 3}


subject_scores = []

print(f"--- Training Personalized AI Models for {len(subjects)} Subjects ---\n")

for subject_id in subjects:
    try:
        
        file_paths = mne.datasets.eegbci.load_data(subject_id, runs)
        raw_files = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in file_paths]
        raw = mne.concatenate_raws(raw_files)
        mne.datasets.eegbci.standardize(raw)
        
        
        raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', skip_by_annotation='edge', verbose=False)
        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=custom_mapping, tmin=-1.0, tmax=4.0, 
                            baseline=None, preload=True, verbose=False)
        
        
        epochs_train = epochs['Left_Fist', 'Right_Fist']
        X = epochs_train.get_data()
        y = epochs_train.events[:, -1]
        
        
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        
        
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        
        
        with mne.utils.use_log_level('ERROR'):
            scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
        
        mean_score = np.mean(scores) * 100
        subject_scores.append(mean_score)
        
        print(f"Subject {subject_id:2d} | Personalized Accuracy: {mean_score:.2f}%")
        
    except Exception as e:
        print(f"Subject {subject_id:2d} | Error: {e}")


print("\n--------------------------------------------------")
print(f"Universal Model Accuracy (Old Method): ~43.44%")
print(f"Personalized Models Average (New Method): {np.mean(subject_scores):.2f}%")
print("--------------------------------------------------")