# %%
# Install from GitHub
# !pip install git+https://github.com/awsaf49/sonics.git

import os
import torch
import librosa
import numpy as np
from sonics import HFAudioClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix

# Restructured model configurations for separate selectors
MODEL_TYPES = ["SpecTTTra-α", "SpecTTTra-β", "SpecTTTra-γ"]
DURATIONS = ["5s", "120s"]

# Mapping for model IDs
def get_model_id(model_type, duration):
    model_map = {
        "SpecTTTra-α-5s": "awsaf49/sonics-spectttra-alpha-5s",
        "SpecTTTra-β-5s": "awsaf49/sonics-spectttra-beta-5s",
        "SpecTTTra-γ-5s": "awsaf49/sonics-spectttra-gamma-5s",
        "SpecTTTra-α-120s": "awsaf49/sonics-spectttra-alpha-120s",
        "SpecTTTra-β-120s": "awsaf49/sonics-spectttra-beta-120s",
        "SpecTTTra-γ-120s": "awsaf49/sonics-spectttra-gamma-120s",
    }
    key = f"{model_type}-{duration}"
    return model_map[key]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model_cache = {}

def load_model(model_type, duration):
    """Load model if not already cached"""
    model_key = f"{model_type}-{duration}"
    if model_key not in model_cache:
        model_id = get_model_id(model_type, duration)
        model = HFAudioClassifier.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        model_cache[model_key] = model
    return model_cache[model_key]


def process_audio(audio_path, model_type, duration):
    """Process audio file and return prediction"""
    try:
        model = load_model(model_type, duration)
        max_time = model.config.audio.max_time

        # Load and process audio
        audio, sr = librosa.load(audio_path, sr=16000)
        chunk_samples = int(max_time * sr)
        total_chunks = len(audio) // chunk_samples
        middle_chunk_idx = total_chunks // 2

        # Extract middle chunk
        start = middle_chunk_idx * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        # Get prediction
        with torch.no_grad():
            chunk = torch.from_numpy(chunk).float().to(device)
            pred = model(chunk.unsqueeze(0))
            prob = torch.sigmoid(pred).cpu().numpy()[0]

        real_prob = 1 - prob
        fake_prob = prob
        
        # Return formatted results
        return {
            "Real": float(real_prob),
            "Fake": float(fake_prob)
        }

    except Exception as e:
        return {"Error": str(e)}


def predict(audio_file, model_type, duration):
    """Gradio interface function"""
    if audio_file is None:
        return {"Message": "Please upload an audio file"}
    return process_audio(audio_file, model_type, duration)

# %%
from tqdm import tqdm
def get_split_mp3(split, folders):
    files = []
    for folder in folders:
        with open(f'/data/{folder}/{split}.txt', 'r') as f:
            folder_files = f.read().splitlines()
            files.extend([f'/data/{folder}/audio/{file}.mp3' for file in folder_files])
    return files

split = 'sample'
folders = ['boomy']
files = get_split_mp3(split, folders)

results = []
y = []

for file in tqdm(files):
    result = process_audio(file, "SpecTTTra-α", "120s")
    results.append(result)
    y.append("Fake")

df = pd.DataFrame(results)
df['y'] = y

# %%
from sklearn.metrics import confusion_matrix

y_pred = df[['Real', 'Fake']].idxmax(axis=1)
y_true = df['y']
# confusion matrix

print(confusion_matrix(y_true, y_pred))
# print it with the labels on the columns and rows
print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

# %%
split = 'sample'
folders = ['suno', 'udio', 'lastfm'] 

# get classification results for sample split (suno and udio are Fake, lastfm is Real)
files = get_split_mp3(split, folders)
results = []
y = []

if not os.path.exists('sonics_results.csv'):
    print('Processing audio files...')
    for file in tqdm(files):
        result = process_audio(file, "SpecTTTra-α", "120s")
        results.append(result)
        if "lastfm" in file:
            y.append("Real")
        else:
            y.append("Fake")
    df = pd.DataFrame(results)
    df['y'] = y

    # add a column with filename
    df['filename'] = files
    # save df as sonics_results.csv
    df.to_csv('sonics_results.csv', index=False)
else:
    print('Loading existing results...')
    df = pd.read_csv('sonics_results.csv')
    y = df['y'].tolist()

# %%
# classification report
from sklearn.metrics import classification_report

y_pred = df[['Real', 'Fake']].idxmax(axis=1)
y_true = df['y']
# classification report with 3 decimals
print(classification_report(y_true, y_pred, digits=3))

# %%
#confusion matrix

source = df['filename'].apply(lambda x: x.split('/')[0])

# confusion matrix where the rows are the true classes (source) and the columns are the predicted classes (Real and Fake)
confusion_matrix(source, y_pred)
# print as a nice latex table with column and row names
print(pd.crosstab(source, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
# normalized without "All" column and row
print(pd.crosstab(source, y_pred, rownames=['True'], colnames=['Predicted'], margins=False, normalize='index'))

# %%
# load 'flask_server/static/transformed_results.csv'
df = pd.read_csv('flask_server/static/transformed_results.csv')

# get all 'classifier' == 'ircamplify'
df_ircamplify = df[df['classifier'] == 'ircamplify']

# make a copy called df_sonics
df_sonics = df_ircamplify.copy()

# %%
# change 'classifier' to 'sonics'
df_sonics['classifier'] = 'sonics'

is_ai = []
confidence = []
pred = []

# for each mp3_path, calculate the prediction with sonics
for idx, row in df_sonics.iterrows():
    result = process_audio(row['mp3_path'], "SpecTTTra-α", "120s")
    if 'Error' in result:
        is_ai.append(None)
        confidence.append(None)
        pred.append(None)
    else:
        is_ai.append(True if result['Fake'] > result['Real'] else False)
        # round to 2 decimals
        confidence.append(round(max(result['Fake'], result['Real'])*100, 2))
        pred.append(round(max(result['Fake'], result['Real']), 2))

    print(f"Added prediction for {row['mp3_path']}")
    print(f"AI: {is_ai[-1]}, Confidence: {confidence[-1]}, Prediction: {pred[-1]}")

df_sonics['is_ai'] = is_ai
df_sonics['confidence'] = confidence
df_sonics['pred'] = pred

# %%
# add df_sonics to the existing df
df = pd.concat([df, df_sonics])
df.to_csv('flask_server/static/transformed_results_2.csv', index=False)
