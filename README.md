# Reg2025 ViseurAI

This repository houses the core components for **ViseurAI** solution, encompassing training, feature extraction, and post-processing functionalities.

---

### Table of Contents

- [Project Structure](#project-structure)
  - [Training](#training)
  - [Feature Extraction](#feature-extraction)
  - [Post-processing](#post-processing)
  - [Similarity Fetching](#similarity-fetching)

---

## Project Structure


### Training

Located in the `Training` folder, this section contains all the code related to our **Custom Decoder**. This includes the implementation of the decoder architecture, training scripts, and configuration files necessary to train the model.

### Feature Extraction

The `FeatureExtraction` folder houses the code responsible for extracting meaningful features from images. We utilize a **Vision Transformer (ViT)** based approach to derive rich, high-dimensional representations of visual data.

### Post-processing

The `Postprocessing.py` script is designed for the self-explanatory task of **post-processing** the data.

### Similarity Fetching

The `fetch_similar.py` script provides functionality to **fetch the closest sentence from the ground truth** dataset.

---
