import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_null_values(df):
    """
    Analyze null values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("=" * 60)
    print("NULL VALUE ANALYSIS")
    print("=" * 60)
    
    # Count null values per column
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    print(f"\nNull values count per column:")
    for col in df.columns:
        if null_counts[col] > 0:
            print(f"  {col}: {null_counts[col]} ({null_percentages[col]:.1f}%)")
    
    print(f"\nTotal rows with at least one null value: {df.isnull().any(axis=1).sum()}")
    print(f"Percentage of rows with null values: {(df.isnull().any(axis=1).sum() / len(df)) * 100:.1f}%")
    
    return null_counts, null_percentages

def remove_null_values_advanced(input_file, output_file, strategy='drop_all'):
    """
    Remove null values from a CSV file using different strategies.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
        strategy (str): Strategy for handling null values
            - 'drop_all': Remove all rows with any null values
            - 'drop_threshold': Remove rows with more than threshold null values
            - 'fill_numeric': Fill numeric columns with mean/median
            - 'fill_categorical': Fill categorical columns with mode
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"\nOriginal dataset shape: {df.shape}")
        
        # Analyze null values
        null_counts, null_percentages = analyze_null_values(df)
        
        # Apply different strategies
        if strategy == 'drop_all':
            df_cleaned = df.dropna()
            print(f"\nStrategy: Remove all rows with any null values")
            
        elif strategy == 'drop_threshold':
            # Remove rows with more than 50% null values
            threshold = len(df.columns) * 0.5
            df_cleaned = df.dropna(thresh=threshold)
            print(f"\nStrategy: Remove rows with more than {threshold:.0f} null values")
            
        elif strategy == 'fill_numeric':
            df_cleaned = df.copy()
            # Fill numeric columns with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    print(f"Filled {col} with median: {median_val:.2f}")
            print(f"\nStrategy: Fill numeric columns with median values")
            
        elif strategy == 'fill_categorical':
            df_cleaned = df.copy()
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df_cleaned[col].isnull().sum() > 0:
                    mode_val = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
                    df_cleaned[col].fillna(mode_val, inplace=True)
                    print(f"Filled {col} with mode: {mode_val}")
            print(f"\nStrategy: Fill categorical columns with mode values")
        
        # Display results
        print(f"\nCleaned dataset shape: {df_cleaned.shape}")
        print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")
        print(f"Percentage of data retained: {(df_cleaned.shape[0] / df.shape[0]) * 100:.2f}%")
        
        # Check if there are still null values
        remaining_nulls = df_cleaned.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"⚠️  Warning: {remaining_nulls} null values still remain")
        else:
            print(f"✅ All null values have been removed")
        
        # Save the cleaned data
        df_cleaned.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def compare_strategies(input_file):
    """
    Compare different strategies for handling null values.
    
    Args:
        input_file (str): Path to the input CSV file
    """
    print("=" * 60)
    print("COMPARING DIFFERENT STRATEGIES")
    print("=" * 60)
    
    df = pd.read_csv(input_file)
    original_rows = len(df)
    
    strategies = ['drop_all', 'drop_threshold', 'fill_numeric', 'fill_categorical']
    results = {}
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        cleaned_df = remove_null_values_advanced(input_file, f"temp_{strategy}.csv", strategy)
        if cleaned_df is not None:
            results[strategy] = {
                'rows_retained': len(cleaned_df),
                'percentage_retained': (len(cleaned_df) / original_rows) * 100,
                'nulls_remaining': cleaned_df.isnull().sum().sum()
            }
    
    # Display comparison
    print(f"\n" + "=" * 60)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Rows Retained':<15} {'% Retained':<12} {'Nulls Remaining':<15}")
    print("-" * 60)
    
    for strategy, result in results.items():
        print(f"{strategy:<20} {result['rows_retained']:<15} {result['percentage_retained']:<12.1f}% {result['nulls_remaining']:<15}")

def main():
    # File paths
    input_file = "CVD Dataset Update.csv"
    output_file = "CVD_Dataset_Cleaned_Advanced.csv"
    
    print("CVD Dataset Null Value Removal Tool")
    print("=" * 60)
    
    # Run basic cleaning
    print("\n1. Basic null removal (drop all rows with nulls):")
    cleaned_df = remove_null_values_advanced(input_file, output_file, 'drop_all')
    
    # Compare strategies
    print("\n2. Comparing different strategies:")
    compare_strategies(input_file)
    
    if cleaned_df is not None:
        print(f"\n✅ Processing completed!")
        print(f"📊 Original rows: {pd.read_csv(input_file).shape[0]}")
        print(f"📊 Cleaned rows: {cleaned_df.shape[0]}")
        print(f"🗑️  Rows removed: {pd.read_csv(input_file).shape[0] - cleaned_df.shape[0]}")
    else:
        print("❌ Failed to process the dataset.")

if __name__ == "__main__":
    main() 