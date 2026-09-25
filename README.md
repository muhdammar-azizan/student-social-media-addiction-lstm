# Student Social Media Addiction LSTM Classifier

LSTM neural network project to classify student social media addiction levels (`Addicted_Score`) based on demographic, social media usage, mental health, and relationship status features. Built for BCS2313 Artificial Intelligence Techniques (UMPSA).

## Objective

Classify each student's addiction score using an LSTM model trained on the [Students Social Media Addiction](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships) dataset from Kaggle, and evaluate the model thoroughly, including overfitting reduction, class imbalance handling, and an explicit accuracy formula check.

## Project Structure

```
.
├── data/
│   └── Students Social Media Addiction.csv   Dataset
├── notebooks/
│   └── SocialMedia_Addiction_Analysis.ipynb  Full analysis and LSTM notebook
├── requirements.txt                          Python dependencies
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

Open the notebook and run the cells in order (or use Run All). The dataset is committed directly in `data/Students Social Media Addiction.csv`, so no manual upload or download step is needed. The notebook loads it directly from `../data/Students Social Media Addiction.csv`, relative to the `notebooks/` folder.

## Models Built

The notebook trains four LSTM models on the same train test split, each building on the one before it.

1. **Set 1 (baseline)**: a single LSTM layer, one Dropout layer, and two Dense layers, trained for 30 epochs with batch size 32. Used as the reference point for every later comparison, and used to first identify overfitting (very high training accuracy, but a noticeably lower and plateauing validation accuracy).
2. **Set 2**: the same architecture as Set 1, trained longer (100 epochs, batch size 64) to confirm and further illustrate the overfitting pattern before any fix is applied.
3. **Improved model**: the same architecture with stronger regularization (Dropout raised to 0.3, L2 regularization on the Dense layer, an extra Dropout layer) and an EarlyStopping callback (monitoring validation loss, patience of 10 epochs, restoring the best weights). This is the model chosen as the best overall model in the notebook.
4. **Class weighted model**: the same architecture and EarlyStopping settings as the improved model, with class weights added during training to address class imbalance for underrepresented Addicted_Score values. The notebook shows honestly that this did not improve prediction for the smallest classes in this dataset, and slightly reduced overall accuracy, mainly because one of the rare classes has zero samples in the training set to begin with.

## Summary of Results

The improved model (EarlyStopping and regularization, no class weighting) is the best performing model in this notebook, reaching a test accuracy of roughly 90 percent, with a much smaller gap between training and validation accuracy than the two baseline runs. The exact figure can vary by about one test sample between separate full runs of the notebook (see Limitations below), and the notebook includes a step by step manual verification, using the trace of the confusion matrix divided by the total number of samples, confirming this accuracy matches scikit learn's own calculation exactly.

## Dataset

- Source: [Students Social Media Addiction, Kaggle (adilshamim8)](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)
- 705 rows and 13 columns, included in this repository at `data/Students Social Media Addiction.csv`

## Limitations

- **Class imbalance at low Addicted_Score values**: the lowest Addicted_Score classes in this dataset have very few samples, in one case only a single sample in the entire dataset. When that sample lands in the test set, the model has zero training examples of that class, so no technique, including class weighting, can fix this without collecting more data or merging classes.
- **Small run to run variation**: even with fixed random seeds (`np.random.seed(42)` and `tf.random.set_seed(42)`) applied to every model, TensorFlow's CPU backend (oneDNN optimized operations) does not guarantee perfectly identical results on every run. Test accuracy for a given model can shift by roughly one test sample (about 0.7 percentage points) between separate full notebook executions. This does not change the overall conclusions, but exact figures should be read as approximate rather than fixed to the decimal point.
