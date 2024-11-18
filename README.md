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

Download and prepare datasets:

```bash
bash data/suno/download_suno.sh
bash data/udio/download_udio.sh
```

## Usage

### Feature Extraction
To extract features using Essentia:

```bash
python scripts/essentia_features.py
```

### Embedding Generation
#### Prepare a checkpoint of CLAP encoder

To use CLAP encoder for conditioning music generation, you have to prepare a pretrained checkpoint file of CLAP.

1. Download a pretrained CLAP checkpoint trained with music dataset (`music_audioset_epoch_15_esc_90.14.pt`)
from the [LAION CLAP repository](https://github.com/LAION-AI/CLAP?tab=readme-ov-file#pretrained-models).
2. Store the checkpoint file to a directory of your choice. (e.g. `./ckpt/clap/music_audioset_epoch_15_esc_90.14.pt`)

Generate embeddings:

```bash
python scripts/get_embed.py
```

### Audio Transformations
Analyze the effects of transformations:

```bash
python src/analyze_audio_transformations.py
```

### Run Flask Server
Start the web interface:

```bash
python flask_server/app.py
```

## File Descriptions

- **data/**: Dataset splits and download scripts.
    - `boomy/`, `lastfm/`, `suno/`, `udio/`: Contain train, validation, test, and sample text files for each dataset.
    - `download_suno.sh`, `download_udio.sh`: Scripts to download Suno and Udio datasets.

- **figures/**: Visualizations of feature distributions.
    - `feature_distributions.pdf`: PDF showing feature distribution plots.

- **flask_server/**: Code for deploying the model via a Flask app.
    - `app.py`: Main Flask application script.
    - `static/graph.json`, `static/graph3d.json`: Precomputed graph data for visualizations.

- **notebooks/**: Jupyter notebooks for analysis and visualization.
    - `add_confound.ipynb`: Adds confounding factors to the dataset.
    - `feature_importance.ipynb`: Analyzes feature importance.
    - `results_audio_transformation.ipynb`: Investigates effects of audio transformations.
    - `umap_visualization.ipynb`: UMAP visualization of embeddings.

- **scripts/**: Helper scripts for feature extraction, metadata retrieval, and dataset management.
    - `essentia_features.py`: Extracts audio features using the Essentia library.
    - `fetch_metadata_song.py`: Fetches metadata for songs.
    - `get_audios.py`: Downloads or processes audio files.
    - `get_embed.py`: Generates audio embeddings.
    - `get_msd.py`: Retrieves Million Song Dataset (MSD) information.
    - `split_train_val_test.py`: Splits data into train, validation, and test sets.
    - `transform_audios.py`: Applies transformations to audio files.

- **src/**: Core analysis and model code.
    - `analyze_audio_transformations.py`: Analyzes impact of audio transformations.
    - `compare_ircamplify.py`: Compares results using IRCAM amplification.
    - `hierarchical_classifier.py`: Implements a hierarchical classifier.

- **utils/**: Utility functions for data and model handling.
    - `data_utils.py`: Data preprocessing utilities.
    - `model_loader.py`: Loads pre-trained models.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
