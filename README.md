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



# Music Recommendation System Metadata Cleaning

## Introduction
This Jupyter Notebook focuses on cleaning and preprocessing metadata for a music recommendation system. Metadata cleaning is essential to ensure accurate and reliable recommendations by removing inconsistencies, errors, and irrelevant information from the dataset. 

## Key Steps
1. *Data Loading*: Load the metadata file containing information about the music tracks, such as title, artist, album, genre, and other relevant attributes.

2. *Data Exploration*: Explore the dataset to identify any inconsistencies, missing values, or anomalies that may affect the quality of recommendations.

3. *Cleaning and Preprocessing*:
   - Handle missing values: Address missing values in the dataset by either imputing them or removing rows with missing information.
   - Standardize data formats: Standardize the format of attributes like genres, artists, and album names to ensure consistency.
   - Remove duplicates: Identify and remove duplicate entries to avoid biasing the recommendation system.
   - Correct errors: Rectify any errors or inconsistencies in the metadata, such as misspelled artist names or incorrect genre labels.

4. *Feature Engineering*:
   - Extract additional features: Extract additional features from the metadata that may enhance the recommendation algorithm's performance, such as artist popularity or track duration.
   - Encoding categorical variables: Encode categorical variables like genres and artists into numerical representations for machine learning algorithms.

5. *Data Export*: Save the cleaned and preprocessed metadata to a new file for further analysis and integration with the recommendation system.

## Conclusion
Metadata cleaning is a crucial step in preparing the data for a music recommendation system. By ensuring the dataset is clean, consistent, and relevant, we lay the foundation for building a robust and accurate recommendation algorithm. This notebook provides a systematic approach to cleaning and preprocessing metadata, setting the stage for the development of an effective music recommendation system.


