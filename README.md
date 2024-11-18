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

### Download prepared dataset

You can download the dataset from: [TODO]

Otherwise, you can create your own by scraping Suno and Udio and calculating the Essentia descriptors and CLAP embeddings.

### Scraping Suno and Udio

To extract refernece metadata from the HTTP request, you can run:

```bash
cd data/suno/
./download_suno.sh
cd ../../
cd data/udio/
./download_udio.sh
cd ../../
```

This will save the metadata of new songs in `data/suno/refs/` and `data/udio/refs/` respectively.

You can then download the audios:

```bash
python scripts/get_audios.py data/suno/
python scripts/get_audios.py data/udio/
```

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

The pre-trained models can be found in: [TODO]

Alternatively, train the hiererchical classifiers and save them:

```bash
python src/hierarchical_classifier.py
```

The models and scalers are saved in `artifacts/models_and_scaler.pkl` and the training classification results in `artifacts/classification_results.pkl`.

They can then be compared with Ircamplify results:

```bash
python src/compare_ircamplify.py
```

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

## File Descriptions

- **data/**: Dataset splits and download scripts.
    - `boomy/`, `lastfm/`, `suno/`, `udio/`: Contain train, validation, test, and sample text files for each dataset.
    - `download_suno.sh`, `download_udio.sh`: Scripts to download Suno and Udio datasets.

- **figures/**: Visualizations of feature distributions.
    - `feature_distributions.pdf`: PDF showing feature distribution plots.

- **flask_server/**: Code for deploying the UMAP visualization via a Flask app.
    - `app.py`: Main Flask application script.
    - `static/graph.json`, `static/graph3d.json`: Precomputed graph data for visualizations.

- **notebooks/**: Jupyter notebooks for analysis and visualization.
    - `add_confound.ipynb`: Adds confounding factor to the dataset to sanity check the hypothesis of the audio transformations.
    - `feature_importance.ipynb`: Analyzes feature importance of Essentia features.
    - `results_audio_transformation.ipynb`: Investigates effects of audio transformations.
    - `umap_visualization.ipynb`: UMAP visualization of embeddings.

- **scripts/**: Helper scripts for feature extraction, metadata retrieval, and dataset management.
    - `essentia_features.py`: Extracts audio features using the Essentia library.
    - `fetch_metadata_song.py`: Fetches metadata for individual songs.
    - `get_audios.py`: Downloads audio files.
    - `get_embed.py`: Generates audio embeddings.
    - `get_msd.py`: Retrieves Million Song Dataset (MSD) information.
    - `split_train_val_test.py`: Splits data into train, validation, and test sets.
    - `transform_audios.py`: Applies transformations to audio files.

- **src/**: Core analysis and model code.
    - `analyze_audio_transformations.py`: Evaluates impact of audio transformations.
    - `compare_ircamplify.py`: Compares results against Ircamplify.
    - `hierarchical_classifier.py`: Implements a hierarchical classifier.

- **utils/**: Utility functions for data and model handling.
    - `data_utils.py`: Data preprocessing utilities.
    - `model_loader.py`: Loads pre-trained models.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
