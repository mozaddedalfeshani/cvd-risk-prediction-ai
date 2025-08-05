# CVD Dataset Null Value Removal Tool

This repository contains Python scripts to remove null values from the CVD (Cardiovascular Disease) dataset using pandas.

## Files

- `remove_null_values.py` - Basic script to remove all rows with null values
- `advanced_null_removal.py` - Advanced script with multiple strategies for handling null values
- `requirements.txt` - Required Python packages
- `CVD Dataset Update.csv` - Original dataset with null values
- `CVD_Dataset_Cleaned.csv` - Cleaned dataset (output from basic script)

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Null Removal

Run the basic script to remove all rows with any null values:

```bash
python remove_null_values.py
```

This will:

- Read the original CSV file
- Remove all rows containing null values
- Save the cleaned data to `CVD_Dataset_Cleaned.csv`
- Display statistics about the cleaning process

### Advanced Null Removal

Run the advanced script for multiple strategies:

```bash
python advanced_null_removal.py
```

This script provides four different strategies:

1. **drop_all**: Remove all rows with any null values
2. **drop_threshold**: Remove rows with more than 50% null values
3. **fill_numeric**: Fill numeric columns with median values
4. **fill_categorical**: Fill categorical columns with mode values

## Results

Based on the analysis of the CVD dataset:

- **Original dataset**: 1,529 rows, 22 columns
- **Null values found**: Multiple columns have missing data
- **Basic cleaning result**: 762 rows retained (49.8% of original data)
- **Rows removed**: 767 rows with null values

## Dataset Information

The CVD dataset contains the following columns:

- Demographic data: Sex, Age
- Physical measurements: Weight, Height, BMI, Abdominal Circumference
- Medical measurements: Blood Pressure, Cholesterol levels, Blood Sugar
- Risk factors: Smoking Status, Diabetes Status, Physical Activity Level
- Family history and risk assessments

## Output Files

- `CVD_Dataset_Cleaned.csv` - Dataset with all null values removed
- `CVD_Dataset_Cleaned_Advanced.csv` - Dataset cleaned using advanced strategies
- Various temporary files for strategy comparison

## Notes

- The basic script removes all rows with any null values, which may result in significant data loss
- The advanced script provides options to preserve more data by using different strategies
- Consider the impact of data loss on your analysis before choosing a strategy
- Always backup your original data before running cleaning scripts
