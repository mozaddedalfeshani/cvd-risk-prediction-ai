# ML Models Workspace

This folder is now organized so a new contributor can quickly understand what is used for data preparation, model development, evaluation, and ready-made artifacts.

## Quick Navigation

| Folder | Purpose | Use this when |
| --- | --- | --- |
| `data/raw/` | Original collected CSV files | You want to inspect the untouched source data |
| `data/processed/` | ML-ready datasets after cleaning and feature engineering | You want to train or evaluate a model |
| `preprocessing/cleaning/` | Cleaning scripts that convert raw data into processed data | You want to regenerate the ML-ready dataset |
| `preprocessing/research/` | Research notes and analysis helpers | You want supporting analysis, not model training |
| `training/experiments/` | Scripts for developing and testing models | You want to build or compare models |
| `training/export/` | Scripts that save deployable `.pkl` files | You want ready-made model artifacts for the app/API |
| `evaluation/scripts/` | Plotting and evaluation utilities | You want charts, comparisons, or dataset visuals |
| `evaluation/reports/` | Generated figures and logs | You want to view saved outputs |
| `models/` | Ready-made trained model files (`.pkl`) | You want to load an already-trained model |

## Start Here

If you want to understand the folder fast, use this order:

1. Read `data/README.md`
2. Read `training/README.md`
3. Read `models/README.md`
4. Read `evaluation/README.md`

## Which File Does What?

### For data cleaning

- `preprocessing/cleaning/clean_mymensing_dataset.py`
  Converts the Mymensing dataset into the processed ML-ready CSV.
- `preprocessing/cleaning/clean_generic_cvd_dataset.py`
  Generic cleaning pipeline for the broader raw CVD dataset.

### For developing models

- `training/experiments/comprehensive_xgboost_training.py`
  Trains the larger comprehensive XGBoost experiment and saves evaluation figures.
- `training/experiments/quick_assessment_ensemble_training.py`
  Trains the smaller quick-assessment ensemble experiment.

### For exporting ready-made models

- `training/export/export_api_xgboost_models.py`
  Exports API-friendly XGBoost model artifacts into `models/`.
- `training/export/export_full_ensemble_model.py`
  Exports the full ensemble model artifact into `models/`.

### For evaluation and visualization

- `evaluation/scripts/plot_training_metrics.py`
  Builds accuracy/loss plots from a saved model artifact.
- `evaluation/scripts/plot_performance_comparison.py`
  Generates a model comparison figure.
- `evaluation/scripts/visualize_dataset.py`
  Creates dataset summary charts from a CSV file.

### For ready-made models

- `models/cvd_full_xgb.pkl`
- `models/cvd_quick_xgb.pkl`
- `models/cvd_quick_model.pkl`
- `models/xgboost_model.pkl`

These are already trained artifacts. You load them for inference or API integration. You do not edit them manually.

## Suggested Workflow

1. Inspect raw data in `data/raw/`
2. Run cleaning scripts from `preprocessing/cleaning/`
3. Train experiments from `training/experiments/`
4. Export deployable artifacts from `training/export/`
5. Review generated results in `evaluation/reports/`
6. Use ready-made `.pkl` files from `models/`
