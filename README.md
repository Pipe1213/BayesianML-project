# Bayesian Machine Learning Project: Review of the paper Loss-Calibrated Approximate Inference in Bayesian Neural Networks (https://arxiv.org/abs/1805.03901).

**Authors:**
1. Carlos Cuevas Villarmin M1-AI
2. Jose Felipe Espinosa Orjuela M1-AI
3. Javier Alejandro Lopetegui González M1-AI

This project applies Bayesian machine learning techniques to classify pneumonia types from chest X-ray images, with a focus on minimizing false negatives for COVID-19 cases. Using a loss calibration approach, we aim to improve model performance on this multi-class classification task. 

The code in this repository is mainly based on the implementation of the paper's authors available in the github repository [https://github.com/AdamCobb/LCBNN](https://github.com/AdamCobb/LCBNN).

## Project Overview
- **Motivation**: Accurate COVID-19 detection is critical to avoid false negatives, and Bayesian methods provide an advantage in estimating uncertainty, which helps improve robustness in medical diagnostics.
- **Approach**: Implemented a utility-based loss function inspired by recent research to optimize the model specifically for COVID-19 detection.

## Key Techniques and Methodology
- **Bayesian Inference**: The BNN framework uses Bayesian inference to estimate uncertainty in predictions, which is valuable in high-stakes medical applications.
- **Loss Calibration**: A customized utility-based loss function is applied to focus the model's performance on minimizing COVID-19 false negatives.
- **Model Architectures**: Two model architectures are explored:
  - **Simple Neural Network**: Baseline model for comparison.
  - **Convolutional Neural Network (CNN)**: Enhanced model to leverage spatial patterns in X-ray images.

## Results
- **Model Performance**: Qualitative assessments indicate improved reliability in COVID-19 classification with calibrated Bayesian approaches.

## Notebooks
1. **experiments_x_rays_images_simple_model.ipynb**: Implements the simple neural network architecture with Bayesian loss calibration.
2. **experiments_x_rays_images_convolutional.ipynb**: Expands on the simple model using a CNN architecture to improve accuracy in image classification.

## Future Work
- **Model Optimization**: Further tuning of hyperparameters and exploration of more advanced Bayesian techniques.

## Instructions for running the code:

1. Install a miniconda distribution compatible with **python-3.6** version (23.10 recommended). An archive of miniconda distributions for any Operative system is availabe in [miniconda-repo](https://repo.anaconda.com/miniconda/).
2. Create a conda environment using the provided environment **Bayesian.yml**: ```conda env create -f bayesian.yml```
3. Activate environment: ```conda activate bayesian```

## Chest X-rays images classification experiments:

The code for the experiments with chest X-rays images for claassifications is based on the experiment done in the paper [On Calibrated Model Uncertainty in Deep Learning](https://arxiv.org/abs/2206.07795). The dataset is built from images availables in [covid-dataset](https://github.com/ieee8023/covid-chestxray-dataset) and [kaggle-Chest-X-ray-images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
