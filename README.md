# Insurance Customer Segmentation using Machine Learning

## Overview
This project applies K-Means clustering to an insurance dataset to group customers using demographic and insurance-related features.

## Dataset
`insurance.csv` contains 1,338 records and these columns:
- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`

## Methods
- Label Encoding for categorical columns
- StandardScaler for feature scaling
- Elbow Method to inspect candidate cluster counts
- K-Means clustering with 4 clusters
- Silhouette Score to evaluate cluster separation
- PCA for 2D visualization

## How to Run
1. Keep `insurance_segmentation.py` and `insurance.csv` in the same folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python insurance_segmentation.py
   ```

The script prints the Silhouette Score and cluster summaries. It also saves the clustered dataset and plots.

## Evaluation Note
K-Means is an unsupervised clustering algorithm, so classification accuracy is not the appropriate metric without ground-truth labels. Use the Silhouette Score and cluster interpretation to assess the result. The reported score can vary if you change preprocessing, features, or the number of clusters.
