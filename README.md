# AI Music Detection
This is the official repository for the paper Detecting AI-Generated Music

### Prepare a checkpoint of CLAP encoder

To use CLAP encoder for conditioning music generation, you have to prepare a pretrained checkpoint file of CLAP.

1. Download a pretrained CLAP checkpoint trained with music dataset (`music_audioset_epoch_15_esc_90.14.pt`)
from the [LAION CLAP repository](https://github.com/LAION-AI/CLAP?tab=readme-ov-file#pretrained-models).
2. Store the checkpoint file to a directory of your choice. (e.g. `./ckpt/clap/music_audioset_epoch_15_esc_90.14.pt`)
