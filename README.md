# Music Genre Classification

Predicting a track's genre from pre-extracted audio features, using the [FMA (Free Music Archive)](https://github.com/mdeff/fma) metadata dataset.

This repo holds my individual contribution — EDA, preprocessing, and modeling on FMA's Librosa-derived audio features — split off from a four-person group final project for UC Berkeley MIDS DATASCI 207, "Music Genre Classification" (Choi, Lee, Sinn, Zhang). The full team project also compared an Echonest feature set and a combined Librosa+Echonest set (see [Related work](#related-work) below); those portions live in teammates' notebooks and aren't included here. The original team repo is private.

## Notebooks

- **[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)** — Exploratory analysis of the feature set: class balance, inter- and intra-feature-group correlation structure, per-feature distributions by genre, mutual information, and dimensionality reduction (PCA + t-SNE) to visualize class separability.
- **[`notebooks/02_preprocessing_modeling.ipynb`](notebooks/02_preprocessing_modeling.ipynb)** — Preprocessing (filtering, splitting, scaling) and modeling: a majority-class baseline, logistic regression, random forest, XGBoost, and a feedforward neural network, each hyperparameter-tuned, plus feature importance via MDI, permutation importance, and SHAP.

## Data

Not included in this repo (large, and not mine to redistribute). The notebooks expect the FMA metadata files one level up from `notebooks/`, at `data/raw/fma_metadata/` (i.e. `../data/raw/fma_metadata/` relative to the notebooks):

- `features.csv` — 518 pre-extracted audio features (MFCC, chroma, spectral contrast/bandwidth/centroid/rolloff, tonnetz, zero-crossing rate, RMSE — each summarized by mean, std, skew, kurtosis, min, max, median)
- `tracks.csv` — track metadata, including top-level genre
- `genres.csv` — genre hierarchy

Get them from the [FMA dataset repo](https://github.com/mdeff/fma) (`fma_metadata.zip`). Expected layout:

```
music_genre_classification/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing_modeling.ipynb
└── data/                      # not included, see above
    └── raw/fma_metadata/
        ├── features.csv
        ├── tracks.csv
        └── genres.csv
```

## Approach

- Dropped tracks with no top-level genre and genres with fewer than 500 tracks, leaving **48,672 tracks across 11 genres** (Rock, Experimental, and Electronic together make up ~70% of the data — the set is heavily imbalanced).
- Stratified 60/20/20 train/val/test split.
- Standardized features; dropped near-constant columns.
- Evaluated primarily on **macro-averaged F1** (rather than accuracy) given the class imbalance, alongside accuracy, ROC-AUC, PR-AUC, and top-3 accuracy.

## Results (test set)

| Model | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Top-3 Acc. |
|---|---|---|---|---|---|
| Majority-class baseline | 29.1% | 0.0410 | 0.500 | 0.091 | 35.1% |
| Logistic Regression | 65.2% | 0.5491 | 0.905 | 0.588 | 88.0% |
| Random Forest | 59.3% | 0.5270 | 0.896 | 0.582 | 82.2% |
| **XGBoost** | **70.9%** | **0.6319** | **0.936** | **0.679** | **90.7%** |
| Feedforward NN | 67.8% | 0.5814 | 0.920 | 0.631 | 89.2% |

Ranking is consistent across every metric: **XGBoost > Neural Net > Logistic Regression > Random Forest** — the same ordering that held on the team's Echonest and combined-feature runs as well.

## Key findings

- Genre separability in raw feature space is limited — PCA/t-SNE show heavily overlapping clusters, especially for Experimental, Folk, Rock, and Electronic.
- Old-Time/Historic is the most reliably classified genre across every model (up to 87–93% recall), likely the most sonically distinctive. Pop and Jazz are the hardest, often confused with Rock and Experimental respectively, across all models — the team's full report suggests this reflects genuine acoustic/label ambiguity in the FMA taxonomy rather than a fixable modeling gap.
- `mfcc_max_04` is the single most important feature by every method tested (MDI, permutation importance, and SHAP) and every model (RF, XGBoost, NN) — an unusually strong point of agreement. SHAP shows it's especially decisive for Old-Time/Historic. Beyond that one feature, importance rankings diverge across methods/models, consistent with the strong intra-MFCC correlation found in EDA diluting and redistributing credit among correlated features.
- Tree ensembles overfit substantially more than the neural net without tuning (RF: ~36pp train/val gap, XGBoost: ~30pp) — dropout gives the NN meaningful built-in regularization that the trees lack until explicitly tuned.

## Related work

The full team project (this repo covers only the FMA/Librosa-features portion) also modeled an 8-feature Echonest set and a combined Librosa+Echonest set on a smaller, 8-genre subset where both sources overlap. XGBoost was the strongest model on every feature set, and the combined features improved every metric for every model, modestly (+0.5 to +3.3pp macro-F1), topping out at 0.734 macro-F1 / 80.8% accuracy on the combined set — the project's best overall result.

## Requirements

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`/`keras`, `shap`, `matplotlib`, `seaborn`, `joblib`
