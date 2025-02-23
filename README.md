# ideal-octo-dollop
Spotify personalized music recommender by mood
# Mood-Based Music Clustering

All input Dataset are available here: https://drive.google.com/drive/folders/1Omn3xpumIaCXVsm-CHdTRBiWQOcgDXyc

This repository contains a project focused on clustering songs by mood using a Spotify dataset. The workflow includes data preprocessing, clustering with KMeans, and a refined labeling approach based on track features (e.g., valence, danceability, energy, acousticness, tempo, loudness).

## 1. Introduction

The goal of this project is to group songs into meaningful mood categories. By analyzing key audio features, we aim to label each cluster in a way that captures the “feel” or “vibe” of the tracks, such as **“Mellow Acoustic,”** **“Powerful Calm,”** **“Upbeat,”** or **“Groovy Dance.”**

## 2. Dataset and Features

- **Dataset Source:**  
  A curated Spotify dataset that includes columns for `valence`, `danceability`, `energy`, `acousticness`, `tempo`, `loudness`, and additional metadata (e.g., track name, artist).

- **Selected Features for Clustering:**  
  - **valence**: Musical positiveness.  
  - **danceability**: Suitability for dancing.  
  - **energy**: Intensity and activity level of a track.  
  - **acousticness**: How acoustic or unplugged a track sounds.  
  - **tempo**: Estimated beats per minute (BPM).  
  - **loudness**: Overall loudness in decibels.

## 3. Data Preprocessing

1. **Normalization & Encoding:**  
   - The data is normalized using a custom function, `normalize_features()`.  
   - Categorical features (if any) are encoded via `encode_categorical_features()`.

2. **Filtering & Scaling:**  
   - Rows with missing values in the key features are dropped.  
   - A `StandardScaler` is applied to the six selected audio features for uniform weighting in clustering.

## 4. Clustering and Evaluation

1. **Determining Optimal Clusters:**  
   - **Elbow Method (SSE)**: Plots the sum of squared errors across different *k* values to find a bend or “elbow.”  
   - **Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters, guiding the choice of *k*.

2. **KMeans Clustering:**  
   - After selecting an optimal *k* (e.g., 4), the model clusters songs into mood-based groups.  
   - The final labels (`Cluster` and `Cluster_Label`) are appended to the DataFrame for easy interpretation.

## 5. Refined Labeling Logic

Based on centroid values (in the original scale) for each cluster, we define a **label_cluster_by_centroid** function that returns descriptive mood labels:

- **Mellow Acoustic:**  
  - High `acousticness` (≥ 0.70)  
  - Implies intimate, unplugged tracks

- **Powerful Calm:**  
  - High `energy` (≥ 0.60) and moderate `valence`  
  - Reflective yet energetic

- **Upbeat:**  
  - High `valence` (≥ 0.55) but lower `danceability` (< 0.40)  
  - Positive mood without a heavy dance element

- **Groovy Dance:**  
  - High `danceability` (≥ 0.70)  
  - Clearly geared toward dance floors

These thresholds can be adjusted to match your dataset’s distribution. Each cluster’s centroid is inspected to decide which label best describes its core attributes.

## 6. Visualization

1. **PCA Plot**  
   - Principal Component Analysis reduces the feature space to two dimensions (`PCA1` and `PCA2`).  
   - A scatter plot shows the distribution of tracks by cluster, and red **X** markers indicate the cluster centers.

2. **Example**  
   ![Clusters Visualization (PCA)]
   Each cluster is colored differently, and the text labels indicate mood-based cluster names.

## 7. Final Output

- **clustered_songs.csv**  
  - A CSV file containing the original track data plus the columns:  
    - `Cluster`: Numeric cluster ID.  
    - `Cluster_Label`: The assigned mood-based label (e.g., “Jazz, Pop”).  
    - `PCA1` and `PCA2`: Principal component coordinates for visualization.

## 8. Repository Structure

- **notebooks/**  
  - Contains the Jupyter notebooks for data cleaning, clustering, and EDA.  
- **scripts/**  
  - Python scripts for data preprocessing, labeling logic, and KMeans modeling.  
- **data/**  
  - The original Spotify dataset (and any other relevant data files).  
- **clustered_songs.csv**  
  - The final CSV output with mood labels.

## 9. How to Run

1. **Clone this repository** and install dependencies (e.g., `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
2. **Open the Jupyter Notebook** (e.g., `mood_clustering.ipynb`) or run the Python script from the command line.
3. **View Results**  
   - The code will generate plots (SSE, silhouette, PCA visualization) and save the final labeled dataset to `clustered_songs.csv`.

## 10. Contributing

Contributions and suggestions are welcome. Feel free to open an issue or submit a pull request for improvements, additional features, or bug fixes.

## 11. License

[MIT License](LICENSE)

---

**Contact:**  
For questions or feedback, reach out to Bilovna Chatterjee [nilovnachatterjee79@gmail.com]
