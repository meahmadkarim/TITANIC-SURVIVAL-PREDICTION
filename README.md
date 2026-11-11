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
