# Student Social Media Addiction — LSTM Classifier

LSTM neural network to classify student social media addiction levels (`Addicted_Score`, 1–9) based on demographic, social media usage, mental health, and relationship-status features. Built for BCS2313 Artificial Intelligence Techniques (UMPSA).

## Objective

Classify each student's addiction score using an LSTM model trained on the [Students Social Media Addiction](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships) dataset from Kaggle.

## Project Structure

```
.
├── data/
│   └── Students Social Media Addiction.csv  # Dataset
├── notebooks/
│   └── SocialMedia_Addiction_Analysis.ipynb # Main analysis & LSTM notebook
├── requirements.txt                         # Python dependencies
└── README.md
```

## Setup

1. Clone this repository.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Notebook

```bash
jupyter notebook notebooks/
```

Open the notebook and run the cells in order. The dataset is committed directly in `data/Students Social Media Addiction.csv`, so no manual upload or download step is needed — the notebook loads it straight from `../data/Students Social Media Addiction.csv` (relative to `notebooks/`).

## Dataset

- Source: [Students Social Media Addiction — Kaggle (adilshamim8)](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)
- 705 rows × 13 columns, included in this repo at `data/Students Social Media Addiction.csv`
