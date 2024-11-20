# Attention is All You Need

This directory contains an implementation of the Transformer model as introduced in the paper "Attention Is All You Need" by Vaswani et al.

## About

Below are a few of the core directories and folders in this repository:

`notebooks/`: Contains Jupyter notebooks that demonstrate the implementation and application of the Transformer model. These notebooks are useful for interactive exploration, visualization, and experimentation with the model's components and training processes.

`scripts/`: Includes various Python scripts designed to facilitate tasks such as data preprocessing, model training, evaluation, and inference. These scripts provide command-line interfaces for executing specific functions related to the Transformer model.

`config.json`: A JSON configuration file that stores hyperparameters and settings for training and evaluating the Transformer model. This file allows for easy modification and management of parameters such as learning rates, batch sizes, and model architecture details.

`checkpoints/`: A directory designated for saving model checkpoints during training. These checkpoints capture the model's state at various stages, enabling the resumption of training from a specific point or the evaluation of the model's performance at different epochs.

`models/`: Contains the definitions and implementations of the model architectures used in this project. This includes the core Transformer model components, such as the encoder, decoder, attention mechanisms, and any custom layers or modules developed for this implementation.

## Getting Started

To begin training and evaluating, follow these:

1. Install the required dependencies by running `pip install -r requirements.txt` from the root directory of this git repository.
2. Run the `notebooks/train.ipynb` notebook to train the Transformer model on a sample dataset.
3. Use the `scripts/evaluation.py` script to evaluate the trained model on a validation set and calculate performance metrics such as BLEU scores.

## References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). _Attention Is All You Need_. arXiv. https://arxiv.org/abs/1706.03762
