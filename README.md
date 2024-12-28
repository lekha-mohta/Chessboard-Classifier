# Chessboard Diagram Classifier

## Overview
This project involves developing a classifier system that processes printed chessboard diagrams from historical chess books and accurately identifies the state of each square on the board. The classifier identifies whether each square is empty or occupied by a chess piece and further classifies the type and color of the piece.

The project includes two modes:
1. **Independent Square Classification**: Each square is classified independently of its position on the board or the state of other squares.
2. **Full-board Classification**: Additional context is considered, including the location of squares and the board's overall state.

## Features
- Dimensionality reduction for image features using Principal Component Analysis (PCA).
- Classification using a k-Nearest Neighbors (k-NN) approach.
- Support for both clean and noisy datasets.

## File Structure
- **system.py**: Contains the core implementation of the dimensionality reduction and classification functions.
- **train.py**: Trains the system on the provided datasets and saves the model data.
- **evaluate.py**: Evaluates the system’s performance on test datasets.
- **utils.py**: Utility functions for reading images and managing model files.

## How It Works
1. **Dimensionality Reduction**: The input images are transformed into feature vectors and reduced to a lower-dimensional space using PCA. This step optimizes the classifier's performance while adhering to the 10-feature constraint.
2. **Classification**:
   - A k-NN classifier (k=6) predicts the label for each square based on the reduced feature vectors.
   - The classifier considers Euclidean distances to find the nearest neighbors and assigns labels based on majority voting.
3. **Full-board Classification**: Incorporates board-level constraints, such as the total number of pieces and maximum limits for each piece type, to improve classification accuracy.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install numpy scipy
   ```

## Usage
1. **Training the Model**:
   ```bash
   python train.py
   ```
   This generates two model files:
   - `model.clean.json.gz`
   - `model.noisy.json.gz`

2. **Evaluating the Model**:
   ```bash
   python evaluate.py
   ```
   This evaluates the system on test datasets and prints the classification accuracy for both square and board modes.

## Input Data
- **Chessboard Images**: JPEG images preprocessed to 400x400 pixels.
- **Labels**: JSON files containing board states with piece positions and types.

## Key Constraints
- The system must use no more than 10 features for classification.
- No third-party libraries are allowed except for `numpy` and `scipy`.
- Model files must not exceed 3MB each.

