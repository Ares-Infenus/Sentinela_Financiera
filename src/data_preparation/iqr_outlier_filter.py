# iqr_outlier_filter.py

import pandas as pd
import os

class IQROutlierFilter:
    """
    Class to detect and remove outliers from a DataFrame using the IQR method.

    Limits are defined as:
        - Lower bound: Q1 - factor * IQR
        - Upper bound: Q3 + factor * IQR
    Where IQR = Q3 - Q1 and the default factor is 1.5.
    """
    
    def __init__(self, factor=1.5, exclude_columns=None):
        """
        Initializes the filter with a specific factor and optional excluded columns.
        
        Args:
            factor (float): Multiplication factor for IQR to define bounds. Default is 1.5.
            exclude_columns (list): List of column names that should not be modified.
        """
        self.factor = factor
        self.exclude_columns = set(exclude_columns) if exclude_columns else set()

    def compute_bounds(self, data, col):
        """
        Computes the lower and upper bounds for the specified column.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data.
            col (str): Column name to evaluate.
        
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - self.factor * iqr
        upper_bound = q3 + self.factor * iqr
        return lower_bound, upper_bound

    def filter_outliers(self, data):
        """
        Filters outliers from all applicable numerical columns in the DataFrame.
        
        Args:
            data (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed from applicable columns.
        """
        filtered_data = data.copy()
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col not in self.exclude_columns:
                lower_bound, upper_bound = self.compute_bounds(data, col)
                filtered_data = filtered_data[(filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)]
        
        return filtered_data

    def export_cleaned_data(self, data, output_path):
        filtered_data = self.filter_outliers(data)
        
        # Get directory name
        dir_name = os.path.dirname(output_path)
        
        # Ensure the directory exists, but only if it's not empty
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save to CSV
        filtered_data.to_csv(output_path, index=False)
        return filtered_data, f"Cleaned data saved to: {output_path}"


# Example usage
if __name__ == "__main__":
    # Sample DataFrame
    df = pd.DataFrame({
        'A': [10, 12, 13, 15, 14, 100, 11, 13, 15, 14],  # Will be filtered
        'B': [5, 6, 7, 8, 9, 10, 11, 12, 200, 6],        # Will be filtered
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]            # Will NOT be filtered
    })
    
    # Specify columns to exclude
    exclude_cols = ['C']

    # Create an instance of the class
    filter_iqr = IQROutlierFilter(factor=1.5, exclude_columns=exclude_cols)
    
    # Define output file path
    output_file = r"filtered_data.csv"  # Change this path as needed

    # Export cleaned data
    filtered_df,message = filter_iqr.export_cleaned_data(df, output_file)
    print(message)
