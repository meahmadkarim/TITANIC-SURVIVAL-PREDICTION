# Titanic Classification Pipeline

A leakage-aware preprocessing and SVM classification exercise.

## Status and provenance

Educational project. The original notebook places this work under a CodSoft internship directory. The maintained `train.py` entry point was added during a portfolio maintenance pass with AI assistance; it is not a claim about the original internship deliverable. Original dataset redistribution rights and any tutorial source still need confirmation.

## Setup and run

Use Python 3.12. From this repository directory:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
python train.py --allow-synthetic-labels
```

The dataset is included in this repository. Use `--data PATH` for another file with the same columns. The script prints evaluation results; no pretrained model or hosted API is supplied. `python train.py --help` lists options.

## Method and limitations

Preprocessing is fitted on training rows only. A fixed seed makes the split reproducible. The included 418-row tested.csv has labels that exactly equal the female-sex indicator. It is not suitable for validating real survival prediction. Default execution refuses these labels; the explicit demo flag is for pipeline illustration only.

The original notebook is retained as a learning artefact. Cleared notebook outputs must be regenerated; use `train.py` for the maintained evaluation path.

## Model, libraries and data flow

The maintained classifier is **scikit-learn `SVC(kernel='linear')`**, a support-vector classifier. A linear kernel does not make this linear regression.

- **pandas** reads the CSV and selects age, fare, sibling/spouse count, parent/child count, passenger class, sex and embarkation port.
- **`train_test_split`** uses a stratified 70/30 split with seed 42.
- Numeric inputs use **`SimpleImputer(strategy='median')`** followed by **`StandardScaler`**. Categorical inputs are imputed and **one-hot encoded** with unseen-category handling.
- **`ColumnTransformer` and `Pipeline`** keep preprocessing fitted on training data rather than the full dataset.
- **`classification_report`** prints precision, recall and F1. These numbers are explicitly labelled demonstration-only when the synthetic-label override is used.

The target guard is essential: the included target exactly matches whether sex is female. Excellent scores on this file demonstrate recovery of that rule, not genuine survival prediction.
