# AI Music Detection
This is the official repository for the paper Detecting AI-Generated Music

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lcrosvila/ai-music-detection.git
cd ai-music-detection
pip install -r requirements.txt
```

## Dataset Preparation

You can find the song IDs for each split in the `data/` directory. To access the corresponding songs, use the following URLs by replacing `{id}` with the actual song ID:

- **Suno**: [`https://suno.com/song/{id}`](https://suno.com/song/{id})  
- **Udio**: [`https://www.udio.com/songs/{id}`](https://www.udio.com/songs/{id})
- 
### Download prepared dataset

You can download the dataset from: [TODO]

You can collect your own and calculate the Essentia descriptors and CLAP embeddings.

### Getting the MSD dataset

To get the subset of MSD songs:

```bash
python scripts/get_msd.py
```

### Feature Extraction
To extract features using Essentia:

```bash
python scripts/essentia_features.py
```

### Embedding Generation
To use CLAP encoder for conditioning music generation, you have to prepare a pretrained checkpoint file of CLAP.

1. Download a pretrained CLAP checkpoint trained with music dataset (`music_audioset_epoch_15_esc_90.14.pt`)
from the [LAION CLAP repository](https://github.com/LAION-AI/CLAP?tab=readme-ov-file#pretrained-models).
2. Store the checkpoint file to a directory of your choice. (e.g. `./ckpt/clap/music_audioset_epoch_15_esc_90.14.pt`)

You can then generate embeddings:

```bash
python get_embed.py -m clap-laion-music -d /data/suno/audio /data/udio/audio -f /path/to/model_file.pt
```

## Usage

### Analyze dataset features

You can perform feature analysis of the Essentia descriptors:

```bash
python notebooks/feature_importance.ipynb
```

And plot the UMAP:

```bash
python notebooks/umap_visualization.ipynb
```

### Train the Hierarchical classifiers

The pre-trained models can be found in: [models_and_scaler.pkl](https://kth-my.sharepoint.com/:u:/g/personal/lcros_ug_kth_se/ETAnWpSVIVNItJPSWD8g3bsBs4oDOnHiwa7eRZRXVqT0zw?e=gvSJv4)

Alternatively, train the hiererchical classifiers and save them:

```bash
python src/hierarchical_classifier.py
```

The models and scalers are saved in `artifacts/models_and_scaler.pkl` and the training classification results in `artifacts/classification_results.pkl`.

They can then be evaluated:

```bash
python src/evaluate_classifier.py
```

Note: If the folder `/data/ircamplify_results/` exists, the predictions of the classifiers will be compared against those from Ircamplify (see `scripts
/ircamplify.py` to query the Ircamplify AI Detector API).

### Performance against transformed audios

Transform the audios:

```bash
python scripts/transform_audios.py
```

And evaluate the classifiers:

```bash
python src/analyze_audio_transformations.py
```

To see the results:

```bash
python notebooks/results_audio_transformation.ipynb
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
