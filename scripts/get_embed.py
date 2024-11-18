from argparse import ArgumentParser
import numpy as np
import torch
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import soundfile
import os
import json

class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int):
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        return wav_data

class CLAPMusic(ModelLoader):
    def __init__(self, model_file: str):
        super().__init__(f"clap-laion-music", 512, 48000)
        self.model_file = model_file

    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.model.load_ckpt(self.model_file)
        self.model.to(self.device)
    
    def _get_embedding(self, ff: str) -> np.ndarray:
        emb = self.model.get_audio_embedding_from_filelist(x=ff, use_tensor=False)
        return emb

class MusiCNN(ModelLoader):
    def __init__(self, input_length=3, input_hop=3):
        super().__init__(f"musicnn", 256, 16000)
        self.input_length = input_length
        self.input_hop = input_hop

    def load_model(self):
        from musicnn.extractor import extractor

        self.model = extractor
    
    def _get_embedding(self, ffs: list) -> list:
        embs = []
        for ff in ffs:
            print('file:', ff)
            aggram, tags1, features = self.model(ff, model='MSD_musicnn', 
                                                 input_length=self.input_length, input_overlap=self.input_hop, extract_features=True)
            emb = features['penultimate']
            embs.append(emb)
        return embs

def get_all_models(model_file):
    ms = [
        CLAPMusic(model_file=model_file),  
        MusiCNN(),      
    ]
    return ms

def main():
    """
    Launcher for caching embeddings of directories using multiple models.
    
    Examples:
    # python get_embed.py -m musicnn -d /data/suno/audio /data/udio/audio
    # python get_embed.py -m clap-laion-music -d /data/suno/audio /data/udio/audio -f /path/to/model_file.pt
    """
    agupa = ArgumentParser(description="Script for extracting embeddings from audio files using various models.")

    # Define the arguments
    agupa.add_argument('-m', '--models', type=str, choices=['musicnn', 'clap-laion-music'], nargs='+', required=True,
                       help="Specify the model(s) to use (e.g., 'musicnn' or 'clap-laion-music').")
    agupa.add_argument('-d', '--dirs', type=str, nargs='+', required=True,
                       help="Specify the directories containing the audio files.")
    agupa.add_argument('-f', '--model-file', type=str, default='./ckpt/clap/music_audioset_epoch_15_esc_90.14.pt',
                       help="Path to the model checkpoint file for CLAP (required if using CLAP model).")
    agupa.add_argument('-w', '--workers', type=int, default=8,
                       help="Number of worker threads for parallel processing.")
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox',
                       help="Path to the SoX binary.")

    args = agupa.parse_args()

    # Get models with the provided model_file argument
    models = {m.name: m for m in get_all_models(args.model_file)}

    for model_name in args.models:
        model = models[model_name]
        model.load_model()
        for d in args.dirs:
            # Create the embeddings directory if it does not exist
            if not os.path.exists(d + '/embeddings/' + model.name):
                os.makedirs(d + '/embeddings/' + model.name)

            mp3s = []
            save_paths = []

            # Collect all mp3 files in the directory
            for mp3 in glob.glob(d + '/*.mp3'):
                npy_path = Path(mp3).parent / 'embeddings' / model.name / (Path(mp3).stem + '.npy')
                if not npy_path.exists():
                    if model.name == 'musicnn':
                        metadata = json.load(open(mp3.replace('.mp3', '.json').replace('audio', 'metadata')))
                        if metadata['duration'] < 3:
                            continue
                    mp3s.append(mp3)
                    save_paths.append(npy_path)

            # Process files in batches
            for i in tqdm(range(0, len(mp3s), args.workers)):
                batch = mp3s[i:i + args.workers]
                batch_save_paths = save_paths[i:i + args.workers]
                embs = model._get_embedding(batch)
                for j, emb in enumerate(embs):
                    np.save(batch_save_paths[j], emb)

if __name__ == "__main__":
    main()
