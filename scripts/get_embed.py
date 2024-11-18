from argparse import ArgumentParser
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import os
import json
from src.model_loader import CLAPMusic, MusiCNN

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
