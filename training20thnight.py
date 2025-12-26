import os
import pickle
import numpy as np
from collections import Counter
from scipy.signal import welch, butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

# --- CONSTANTS ---
SAMPLING_RATE = 128
BANDS = {
    'Delta': [1, 4], 'Theta': [4, 8], 'Alpha': [8, 13],
    'Beta': [13, 30], 'Gamma': [30, 45]
}

# --- FUNCTIONS ---
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def get_band_power(data, band, fs):
    freqs, psd = welch(data, fs, nperseg=fs)
    idx_band = np.logical_and(freqs >= BANDS[band][0], freqs <= BANDS[band][1])
    return np.sum(psd[idx_band])

# --- LOAD DATA ---
print("Loading DEAP dataset and extracting features...")

data_folder_path = r"C:\Major Project\DEAP_dataset"
all_features, all_labels = [], []

all_filenames = [f for f in os.listdir(data_folder_path) if f.endswith('.dat')]
print(f"Found {len(all_filenames)} data files.")

for filename in all_filenames:
    file_path = os.path.join(data_folder_path, filename)
    with open(file_path, 'rb') as f:
        subject_data = pickle.load(f, encoding='latin1')

    eeg_data = subject_data['data']
    labels = subject_data['labels']

    for trial in range(40):
        channel_data = eeg_data[trial, 0, :]
        filtered_data = bandpass_filter(channel_data, 1, 45, SAMPLING_RATE)

        features = [get_band_power(filtered_data, band, SAMPLING_RATE) 
                    for band in BANDS]
        all_features.append(features)

        # 4-class labels
        val, aro = labels[trial][0], labels[trial][1]

        if aro >= 5 and val >= 5:
            emotion = "happy"
        elif aro < 5 and val >= 5:
            emotion = "calm"
        elif aro < 5 and val < 5:
            emotion = "sad"
        else:
            emotion = "angry"

        all_labels.append(emotion)

X = np.array(all_features)
y = np.array(all_labels)

print("Original Label Distribution:", Counter(y))

# --- SEPARATE BY CLASS ---
X_list = list(X)
y_list = list(y)

happy = [X_list[i] for i in range(len(y_list)) if y_list[i] == "happy"]
calm  = [X_list[i] for i in range(len(y_list)) if y_list[i] == "calm"]
sad   = [X_list[i] for i in range(len(y_list)) if y_list[i] == "sad"]
angry = [X_list[i] for i in range(len(y_list)) if y_list[i] == "angry"]

# --- NEW TARGET COUNTS ---
TARGET_HAPPY = 2000
TARGET_CALM  = 2000
TARGET_SAD   = 200
TARGET_ANGRY = 100

# --- OVERSAMPLE / UNDERSAMPLE ---
happy = resample(happy, replace=True, n_samples=TARGET_HAPPY, random_state=42)
calm  = resample(calm,  replace=True, n_samples=TARGET_CALM,  random_state=42)
sad   = resample(sad,   replace=True, n_samples=TARGET_SAD,   random_state=42)
angry = resample(angry, replace=True, n_samples=TARGET_ANGRY, random_state=42)

# --- REBUILD DATASET ---
X_new = happy + calm + sad + angry
y_new = (["happy"] * TARGET_HAPPY) + \
        (["calm"]  * TARGET_CALM)  + \
        (["sad"]   * TARGET_SAD)   + \
        (["angry"] * TARGET_ANGRY)

X = np.array(X_new)
y = np.array(y_new)

print("New Label Distribution:", Counter(y))

# --- TRAIN MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining model...")
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Training Accuracy: {acc * 100:.2f}%")

# --- SAVE MODEL ---
joblib.dump(model, "emotion_model.pkl")


print("\nModel saved as 'emotion_model.pkl'")
