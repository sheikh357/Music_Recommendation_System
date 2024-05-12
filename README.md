# Music_Recommendation_System

# Audio Feature Extraction README

## Introduction

This Python script facilitates the extraction of audio features from a collection of audio files using the Librosa and Pydub libraries. These features are crucial for various audio analysis tasks such as music genre classification, speech recognition, and sound event detection.

## Key Steps

### 1. Loading and Preprocessing

Audio files are loaded using Pydub and normalized to prepare them for feature extraction using Librosa.

### 2. Feature Extraction

The extract_features function computes Mel-frequency cepstral coefficients (MFCCs), spectral centroid, and zero-crossing rate for each audio file.

### 3. Processing Directory

The process_directory function iterates through all files within a directory, extracting features for each valid audio file encountered.

### 4. Parallel Processing

Multiprocessing is employed to distribute the feature extraction tasks across multiple CPU cores, significantly reducing processing time, especially for large-scale datasets.

### 5. Normalization and Dimensionality Reduction

Extracted features are standardized using StandardScaler and then subjected to Principal Component Analysis (PCA) for dimensionality reduction, ensuring computational efficiency while retaining relevant information.

### 6. Data Export

The reduced features, along with folder and file names, are saved to a CSV file named extracted_features_selected_folders.csv, facilitating further analysis and machine learning model training.






