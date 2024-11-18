import essentia
import essentia.standard as es
import json
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import librosa

def calculate_dc_drift_windows(y, sr, window_size=0.1):
    """
    Calculate the DC drift (offset) of an audio signal over windows of specified size.
    """
    window_samples = int(window_size * sr)
    num_windows = len(y) // window_samples
    dc_drifts = []
    
    for i in range(num_windows):
        start = i * window_samples
        end = (i + 1) * window_samples
        window = y[start:end]
        dc_drift = np.abs(np.mean(window))
        dc_drifts.append(dc_drift)
    
    return float(np.mean(dc_drifts)), float(np.std(dc_drifts))

# Function to extract features and save them to a JSON file
def process_file(args):
    audio_path, features_path = args

    if features_path.endswith('.json') and os.path.exists(features_path):
        # if 'lowlevel.dc_drift.mean' and 'lowlevel.dc_drift.stdev' are already in the file, skip
        try:
            with open(features_path, 'r') as f:
                features_dict = json.load(f)
        except:
            print('error loading file', features_path)

        if 'lowlevel.dc_drift.mean' in features_dict and 'lowlevel.dc_drift.stdev' in features_dict:
            return
        else:
            # load the audio and calculate the DC drift
            y, sr = librosa.load(audio_path, sr=None)
            dc_drift_mean, dc_drift_std = calculate_dc_drift_windows(y, sr)

            features_dict['lowlevel.dc_drift.mean'] = dc_drift_mean
            features_dict['lowlevel.dc_drift.stdev'] = dc_drift_std

            # if the folder does not exist, create it
            os.makedirs(os.path.dirname(features_path), exist_ok=True)
            
            # Save the features to a JSON file
            with open(features_path, 'w') as f:
                json.dump(features_dict, f, indent=4)
            return
        
    # Extract features using Essentia's MusicExtractor
    try:
        features, features_frames = es.MusicExtractor(
            lowlevelStats=['mean', 'stdev'], 
            rhythmStats=['mean', 'stdev'], 
            tonalStats=['mean', 'stdev']
        )(audio_path)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return
    
    # Prepare the dictionary to store features
    features_dict = {}
    for feature in features.descriptorNames():
        if isinstance(features[feature], np.ndarray):
            features_dict[feature] = features[feature].tolist()
        else:
            features_dict[feature] = features[feature]
    
    # load the audio and calculate the DC drift
    y, sr = librosa.load(audio_path, sr=None)
    dc_drift_mean, dc_drift_std = calculate_dc_drift_windows(y, sr)

    features_dict['lowlevel.dc_drift.mean'] = dc_drift_mean
    features_dict['lowlevel.dc_drift.stdev'] = dc_drift_std

    # if the folder does not exist, create it
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    
    # Save the features to a JSON file
    with open(features_path, 'w') as f:
        json.dump(features_dict, f, indent=4)

# List of folders to process
folders = [
    '/data/suno', 
    '/data/udio', 
    '/data/lastfm'
]

# Iterate over each folder with progress bar
for folder in tqdm(folders, desc="Processing folders"):
    audio_files = []

    for split in ['train', 'val', 'test']:
        with open(f'{folder}/{split}.txt', 'r') as f:
            audio_files += f.read().splitlines()    
    
    # Prepare file paths for processing
    file_paths = [
        (os.path.join(folder, 'audio', file + '.mp3'), 
         os.path.join(folder, 'essentia_features', file + '.json')) 
        for file in audio_files
    ]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, file_paths), 
                  total=len(file_paths), desc=f"Processing files in {folder}", leave=False))
