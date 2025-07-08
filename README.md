# Learning Fast to Detect Slow: A Few-Shot Neural Approach to Slow DoS Attack Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the research paper "Learning Fast to Detect Slow: A Few-Shot Neural Approach to Slow DoS Attack Detection". The proposed ENE4 (Ensemble Network of feature Extractors for Few-shot learning) framework tackles the challenge of detecting stealthy, low-rate Denial-of-Service (DoS) attacks by integrating supervised and unsupervised learning under data-scarce conditions.

## Key Features

* **Few-Shot Learning for Cybersecurity:** Capable of detecting underrepresented and evasive Slow DoS attacks with minimal labeled data;
* **Hybrid DL Architecture:** Combines a pre-trained supervised MLP and a Variational Autoencoder (VAE) to extract both attack-specific and attack-agnostic features;
* **Knowledge Transfer:** Uses a transfer learning approach to adapt knowledge from frequent attack types to stealthier ones;
* **Data Augmentation and Focal Loss:** Enhances robustness and handles class imbalance;
* **Benchmark-Ready:** Evaluated on CIC-IDS2017 and 5G-NIDD datasets, outperforming classical ML baselines under few-shot constraints.

## Repository Structure

* **ene4/:**
  * main_ene4.py: Main script for training and evaluating the ENE4 framework;
  * vae_module.py: Contains the implementation of the Variational Autoencoder used for unsupervised feature extraction;
  * mlp_module.py: Pre-trained supervised MLP for attack-oriented feature extraction;
  * classifier_head.py: Combines features and performs final classification;
  * data_utils.py: Utilities for loading, preprocessing, and augmenting datasets.

## Requirements

* Python (3.10+)
* PyTorch (2.2.1)
* NumPy (1.26.4)
* Scikit-learn (1.4.1.post1)
* pandas (2.2+)
* tqdm (4.66+)

## Citation
```
Coming soon...
```

## Acknowledgment
This work was supported by the EU-funded projects FAIR (PE00000013) and SERICS (PE00000014) under the NextGeneration EU and NRRP MUR programs.
