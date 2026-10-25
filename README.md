# Galaxy Detector Neural Network

## Streamlit App

[Galaxy Detector](https://galaxy-detector-neural-network-zgyjmg4jrhfrgu6y2iftmq.streamlit.app/)

## Task

**Galaxy detection + morphology classification**

- Input: takes an image (grayscale or RGB)
- Output 1: galaxy present vs. no galaxy present
- Output 2: if a galaxy, its morphology: elliptical, spiral, barred spiral, lenticular, edge-on disk, or irregular

## Data

- **Galaxies:** [Galaxy Zoo 2](https://github.com/mwalmsley/galaxy-datasets) (`data/gz2`). Morphology classes are derived from the volunteer votes in [cnn/morphology.py](cnn/morphology.py). GZ2 has no explicit lenticular or irregular labels, so these two are approximations:
  - lenticular: a disk with no bar and no spiral arms (S0-like)
  - irregular: mostly voted "something odd" (this also catches rings, lenses and dust lanes)
- **Non-galaxies:** GZ2 images voted as stars/artifacts, [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) photos (downloaded automatically), the photos in `data/images/non_galaxies`, and a little random noise

## Architecture

**Convolutional Neural Network (CNN)** with a shared backbone and two heads

```
Input image (128x128x3 RGB)
↓
4 x [Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPooling]  (32, 64, 128, 256 filters)
↓
Global average pooling → Dropout
↓                                  ↓
Detect head: Dense (2)             Morphology head: Dense (128) → ReLU → Dropout → Dense (6)
```

**Loss:** CrossEntropyLoss for detection + class-weighted CrossEntropyLoss for morphology (only labelled galaxies count)

**Optimizer:** AdamW with warmup + cosine learning-rate schedule

## Usage

```
pip install -r requirements.txt        # app only
pip install -r requirements-train.txt  # training (see the file for the CUDA build of torch)

python -m cnn.preresize                # optional, once: cropped 160px copies of the GZ2 images (much faster training)
python -m cnn.handler                  # train; saves the best checkpoint to models/galaxy_morph_cnn.pth
python -m cnn.handler --subset 3000 --epochs 2   # quick smoke test
streamlit run app.py
```

Training metrics are tracked in the [Weights & Biases](https://wandb.ai) project `galaxy-detector-neural-network`. Put `WANDB_API_KEY=...` in `.env`, or pass `--wandb-mode offline` / `--wandb-mode disabled`.

## Tools & Methodology

### Original implementation (by me)

I wrote the first versions of this project by hand to learn how neural networks work from the ground up.

- **MLP from scratch** ([`mlp/`](mlp/)): a fully connected network written in pure NumPy, with no ML framework. I implemented the forward pass, sigmoid activations, binary cross-entropy loss, backpropagation, mini-batch gradient descent, training/validation metrics and plotting.
- **Data tooling:** an OpenCV image preprocessor with augmentation, a dataset loader, and an image extractor that collects galaxy images from the NASA Image & Video Library and random non-galaxy photos from Lorem Picsum.
- **First CNN** ([`cnn/`](cnn/)): a PyTorch convolutional network for binary galaxy detection, trained on Galaxy Zoo 2, with dataset wrappers, a training handler, and the Streamlit app for uploading and classifying images.

### AI-assisted acceleration (Claude Opus 5.5)

I then used **Claude Opus 5.5** in [Claude Code](https://claude.com/claude-code) as a pair programmer to review the CNN and extend it to morphology classification. I made the design decisions (the taxonomy source, keeping a custom CNN rather than a pretrained backbone, which negative data to use, and how to split the requirements files), and reviewed and approved the plan before any code was written. Claude helped with:

- **Code review:** finding bugs in the original CNN, such as the missing activations between conv layers, random-noise "non-galaxy" images, and accuracy measured on the training set
- **Morphology expansion:** deriving 6 morphology classes from Galaxy Zoo 2 votes and redesigning the network with a shared backbone and separate detection and morphology heads
- **Data pipeline:** RGB input, augmentation, real non-galaxy datasets, stratified train/val/test splits, class balancing, and image pre-resizing for faster training
- **Training improvements:** AdamW with a warmup + cosine schedule, mixed precision, best-checkpoint saving with early stopping, and per-class evaluation reports
- **Tooling:** W&B experiment tracking and dependency-conflict cleanup

### Tools

- **Language & ML:** Python, NumPy, PyTorch, torchvision
- **Data:** [galaxy-datasets](https://github.com/mwalmsley/galaxy-datasets) (Galaxy Zoo 2), Caltech-101, OpenCV, Pillow, pandas
- **App:** Streamlit (Streamlit Community Cloud)
- **Experiment tracking:** Weights & Biases
- **AI assistance:** Claude Code with Claude Opus 5.5

## Resources

- [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo)
- [NASA Image & Video Library](https://images.nasa.gov/)
- [Lorem Picsum](https://picsum.photos/)
- [PyTorch Galaxy Datasets](https://github.com/patrikasvanagas/pytorch-galaxy-datasets)

## Roadmap

- [x] Write image preprocessor
- [x] Write dataset loader
- [x] Write neural network
  - [x] Forward pass
  - [x] Loss function
  - [x] Backpropagation
  - [x] Gradient descent
  - [x] Training loop
  - [x] Loss tracking
- [x] Write image extractor
- [x] Data augmentation
- [x] Convert to Convolutional Neural Network (CNN)
- [x] Curate larger training set
- [x] Improve training speed
- [x] Build simple UI for image upload then prediction
- [x] Expand network to classify galaxy types
- [x] Add W&B Project
