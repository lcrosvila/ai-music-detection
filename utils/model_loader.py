from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
import soundfile

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
    def __init__(self):
        super().__init__(f"clap-laion-music", 512, 48000)
        self.model_file = '/home/laura/aimir/embeddings_env/lib/python3.11/site-packages/fadtk/.model-checkpoints/music_audioset_epoch_15_esc_90.14.pt'

    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.model.load_ckpt(self.model_file)
        self.model.to(self.device)
    
    # the input is the file
    def _get_embedding(self, ff: str) -> np.ndarray:
        emb = self.model.get_audio_embedding_from_filelist(x = ff, use_tensor=False)
        return emb
    
    def _get_embedding_from_data(self, audio: np.ndarray) -> np.ndarray:
        emb = self.model.get_audio_embedding_from_data(x = audio, use_tensor=False)
        return emb

class MusiCNN(ModelLoader):
    def __init__(self, input_length = 3, input_hop = 3):
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
