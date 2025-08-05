import pandas as pd
import numpy as np

def remove_null_values(input_file, output_file):
    """
    Remove null values from a CSV file using pandas.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Display original data info
        print(f"\nOriginal dataset shape: {df.shape}")
        print(f"Original dataset columns: {list(df.columns)}")
        print(f"Original null values count:")
        print(df.isnull().sum())
        
        # Remove rows with any null values
        df_cleaned = df.dropna()
        
        # Display cleaned data info
        print(f"\nCleaned dataset shape: {df_cleaned.shape}")
        print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")
        print(f"Percentage of data retained: {(df_cleaned.shape[0] / df.shape[0]) * 100:.2f}%")
        
        # Save the cleaned data to a new CSV file
        df_cleaned.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
        
        # Display summary statistics
        print(f"\nSummary of cleaned data:")
        print(df_cleaned.describe())
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    # File paths
    input_file = "CVD Dataset Update.csv"
    output_file = "CVD_Dataset_Cleaned.csv"
    
    # Remove null values
    cleaned_df = remove_null_values(input_file, output_file)
    
    if cleaned_df is not None:
        print(f"\n✅ Successfully cleaned the dataset!")
        print(f"📊 Original rows: {pd.read_csv(input_file).shape[0]}")
        print(f"📊 Cleaned rows: {cleaned_df.shape[0]}")
        print(f"🗑️  Rows removed: {pd.read_csv(input_file).shape[0] - cleaned_df.shape[0]}")
    else:
        print("❌ Failed to clean the dataset.")

if __name__ == "__main__":
    main() 