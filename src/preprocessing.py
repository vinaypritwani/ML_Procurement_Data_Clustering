import pandas as pd  
# Import the pandas library for data manipulation and analysis.

def load_data():
    """
    Loads procurement data and taxonomy data from Excel files using absolute paths.
    Prints the first few rows of each dataset to verify structure.
    Returns two DataFrames: df_procurement and df_taxonomy.
    """
    # Replace these paths with the exact location of your Excel files:
    procurement_file = r"C:\Users\Vinay Pritwani\Desktop\procurement_ml_project\data\Vinay J.xlsx"
    # The 'r' before the string makes it a raw string, which is useful for Windows paths.
    taxonomy_file = r"C:\Users\Vinay Pritwani\Desktop\procurement_ml_project\data\categories_taxonomy.xlsx"
    
    # Print out the absolute paths for debugging.
    import os
    print("Procurement file (absolute):", os.path.abspath(procurement_file))
    print("Exists?", os.path.exists(procurement_file))
    print("-" * 50)

    # Read the Excel files into DataFrames.
    df_procurement = pd.read_excel(procurement_file)
    df_taxonomy = pd.read_excel(taxonomy_file)

    # Print the first 5 rows of each DataFrame to verify the data.
    print("=== Procurement Data (first 5 rows) ===")
    print(df_procurement.head(), "\n")
    print("=== Category Taxonomy (first 5 rows) ===")
    print(df_taxonomy.head(), "\n")

    # Return the DataFrames for further processing.
    return df_procurement, df_taxonomy

if __name__ == "__main__":
    # When this script is run directly, call load_data() to load and display the data.
    load_data()

