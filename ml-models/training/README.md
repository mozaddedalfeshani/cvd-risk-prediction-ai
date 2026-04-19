# Training Folder

## Purpose

Everything in this folder is for building models, not for loading ready-made models.

## Structure

- `experiments/`
  Model development and training scripts.
- `export/`
  Scripts that save trained artifacts into `../models/`.

## Main Files

- `experiments/comprehensive_xgboost_training.py`
  Larger experiment using the full processed feature set.
- `experiments/quick_assessment_ensemble_training.py`
  Smaller experiment using a reduced feature set.
- `export/export_api_xgboost_models.py`
  Produces XGBoost artifacts for API use.
- `export/export_full_ensemble_model.py`
  Produces the full ensemble artifact.

## Rule of Thumb

If you want to develop or retrain a model, start here.
If you only want an already-trained model, go to `../models/`.
