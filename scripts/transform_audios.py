import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from src.model_loader import CLAPMusic
import soundfile as sf

model_file = './ckpt/clap/music_audioset_epoch_15_esc_90.14.pt'
model = CLAPMusic(model_file=model_file)
model.load_model()

def get_split_mp3(split, folders):
    files = []
    for folder in folders:
        with open(f'/data/{folder}/{split}.txt', 'r') as f:
            folder_files = f.read().splitlines()
            files.extend([f'/data/{folder}/audio/{file}.mp3' for file in folder_files])
    return files

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_low_pass_filter(audio, sr, audio_path, cutoff=5000, order=5, with_mp3=False):
    """
    Apply low-pass filter to audio and either save as MP3 or compute embeddings.
    
    Args:
        audio: Input audio data
        sr: Sample rate
        audio_path: Path to the original audio file
        cutoff: Filter cutoff frequency in Hz (default: 5000)
        order: Filter order (default: 5)
        with_mp3: If True, saves as MP3 file (default: False)
    
    Returns:
        Either the path to the saved MP3 file or the computed embeddings
    """
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    
    if with_mp3:
        save_file_mp3 = f'/data/{folder}/audio/transformed/low_pass_{cutoff}/{file}_low_pass_{cutoff}.mp3'

    save_file = f'/data/{folder}/audio/transformed/low_pass_{cutoff}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    
    # Apply low-pass filter
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = filtfilt(b, a, audio)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Compute and save embeddings
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)

    if with_mp3:
        # Save as MP3
        sf.write(save_file_mp3, y, sr)
        return save_file_mp3
    else:
        return emb


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_high_pass_filter(audio, sr, audio_path, cutoff=5000, order=5, with_mp3=False):
    """
    Apply high-pass filter to audio and either save as MP3 or compute embeddings.
    
    Args:
        audio: Input audio data
        sr: Sample rate
        audio_path: Path to the original audio file
        cutoff: Filter cutoff frequency in Hz (default: 5000)
        order: Filter order (default: 5)
        with_mp3: If True, saves as MP3 file (default: False)
    
    Returns:
        Either the path to the saved MP3 file or the computed embeddings
    """
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    
    if with_mp3:
        save_file_mp3 = f'/data/{folder}/audio/transformed/high_pass_{cutoff}/{file}_high_pass_{cutoff}.mp3'

    save_file = f'/data/{folder}/audio/transformed/high_pass_{cutoff}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    
    # Apply high-pass filter
    b, a = butter_highpass(cutoff, sr, order=order)
    y = filtfilt(b, a, audio)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Compute and save embeddings
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)

    if with_mp3:
        # Save as MP3
        sf.write(save_file_mp3, y, sr)
        return save_file_mp3
    else:
        return emb

def add_noise(audio, audio_path, noise_factor=0.005):
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/data/{folder}/audio/transformed/noise_{str(noise_factor).replace(".", "_")}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        noise = np.random.randn(len(audio))
        y = audio + noise_factor * noise
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)
    return emb

def decrease_sample_rate(audio, sr, audio_path, target_sr=8000, with_mp3=False):
    """
    Decrease the sample rate of an audio file and either save as MP3 or compute embeddings.
    
    Args:
        audio: Input audio data
        sr: Original sample rate
        audio_path: Path to the original audio file
        target_sr: Target sample rate (default: 8000)
        with_mp3: If True, saves as MP3 file (default: False)
    
    Returns:
        Either the path to the saved MP3 file or the computed embeddings
    """
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    
    if with_mp3:
        save_file_mp3 = f'/data/{folder}/audio/transformed/decrease_sr_{target_sr}/{file}_decrease_sr_{target_sr}.mp3'

    save_file = f'/data/{folder}/audio/transformed/decrease_sr_{target_sr}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    
    # Resample the audio
    y = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Compute and save embeddings
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)

    if with_mp3:
        # Save as MP3
        sf.write(save_file_mp3, y, target_sr)
        return save_file_mp3
    else:
        return emb

def add_tone(audio, sr, audio_path, tone_freq=10000, tone_db=3):
    # Add a tone to an audio file at a specified frequency and amplitude.
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/data/{folder}/audio/transformed/sine_{tone_freq}_{tone_db}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        # Generate time array
        t = np.arange(len(audio)) / sr
        tone = np.sin(2 * np.pi * tone_freq * t)
        amplitude_factor = 10 ** (tone_db / 20)
        
        # Normalize the tone to match the amplitude of the original audio
        max_amplitude = np.max(np.abs(audio))
        tone = tone * max_amplitude * amplitude_factor
        
        # Add the tone to the original audio
        y_with_tone = audio + tone
        
        # Normalize the result to prevent clipping
        y_with_tone = y_with_tone / np.max(np.abs(y_with_tone))

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y_with_tone])[0]
    np.save(save_file, emb)
    return emb

def add_dc_drift(audio, sr, audio_path, drift_amount, with_mp3=False):
    """
    Add DC drift to audio and either save as MP3 or compute embeddings.
    
    Args:
        audio: Input audio data
        sr: Sample rate
        audio_path: Path to the original audio file
        drift_amount: Amount of DC drift to add
        with_mp3: If True, saves as MP3 file (default: False)
    
    Returns:
        Either the path to the saved MP3 file or the computed embeddings
    """
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    drift_str = str(drift_amount).replace(".", "_")
    
    if with_mp3:
        save_file_mp3 = f'/data/{folder}/audio/transformed/dc_drift_{drift_str}/{file}_dc_drift_{drift_str}.mp3'

    save_file = f'/data/{folder}/audio/transformed/dc_drift_{drift_str}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    
    # Add DC drift
    y = audio + drift_amount
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Compute and save embeddings
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)

    if with_mp3:
        # Save as MP3
        sf.write(save_file_mp3, y, sr)
        return save_file_mp3
    else:
        return emb

def main():
    split = 'test'
    folders = ['suno', 'udio', 'lastfm'] 

    cutoffs = [100, 500, 1000, 3000, 5000, 8000, 10000, 12000, 16000, 20000]
    sr_decrease = [8000, 16000, 22050, 24000, 44100]

    # cutoffs = [1000, 8000, 10000, 16000]
    # sr_decrease = [22050, 24000]

    files = get_split_mp3(split, folders)

    for folder in folders:
        if not os.path.exists(f'/data/{folder}/audio/transformed'):
            os.makedirs(f'/data/{folder}/audio/transformed')

    for file in tqdm(files):
        audio, sr = librosa.load(file, sr=None)
        for cutoff in cutoffs:
            apply_low_pass_filter(audio, sr, file, cutoff=cutoff, with_mp3=False)
            apply_high_pass_filter(audio, sr, file, cutoff=cutoff, with_mp3=False)
        for sr_target in sr_decrease:
            decrease_sample_rate(audio, sr, file, target_sr=sr_target, with_mp3=False)
    
    print("All files processed.")

        
if __name__ == '__main__':
    main()
