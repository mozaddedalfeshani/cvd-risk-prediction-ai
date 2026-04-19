# Models Folder

## Purpose

This folder stores ready-made trained model artifacts.

## What belongs here

- `.pkl` files that are already trained and ready to load
- Nothing else

## Current Artifacts

- `cvd_full_xgb.pkl`
  Full-feature XGBoost artifact.
- `cvd_quick_xgb.pkl`
  Quick-assessment XGBoost artifact.
- `cvd_quick_model.pkl`
  Quick ensemble artifact.
- `xgboost_model.pkl`
  Saved XGBoost artifact used by evaluation scripts.

## Rule of Thumb

If you want to load a model, open this folder.
If you want to train a model, go to `../training/`.
