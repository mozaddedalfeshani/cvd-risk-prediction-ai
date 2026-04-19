# Data Folder

## Purpose

This folder separates untouched source data from model-ready data.

## Structure

- `raw/`
  Original CSV files collected from source.
- `processed/`
  Cleaned and feature-engineered datasets used for ML experiments.

## Main Files

- `raw/MymensingUniversity.csv`
  Main raw dataset collected for this project.
- `raw/Raw_Dataset.csv`
  Raw CVD dataset copy/reference dataset.
- `processed/MymensingUniversity_ML_Ready.csv`
  Primary dataset used for training and evaluation.

## Rule of Thumb

If you want to inspect what was collected, open `raw/`.
If you want to train a model, use `processed/`.
