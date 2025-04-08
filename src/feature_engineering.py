import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Cleans the DataFrame by filling missing values with empty strings.
    """
    df['invoice_description'] = df['invoice_description'].fillna("") # Replace NaN values in supplier names with an empty string.
    return df

def vectorize_text(df,column,max_features=1000):
    """
    Converts text data into numerical features using TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer(stop_words='english',max_features=max_features)
      # Apply the TF-IDF transformation on the selected text column.
    text_features = vectorizer.fit_transform(df[column])
    text_df = pd.DataFrame(text_features.toarray(),columns=[f"{column}_tfidf_{i}" for i in range(text_features.shape[1])])
    # Convert the sparse matrix into a Pandas DataFrame with readable column names.
    return text_df, vectorizer


def prepare_features(df):
    """
    Prepares a feature matrix by cleaning data and applying TF-IDF vectorization.
    """
    df = clean_data(df)  # First, clean the data.

    gl_features, gl_vectorizer = vectorize_text(df, 'invoice_description')  
    # Convert 'gl_description' text into numerical features.

    supplier_features, supplier_vectorizer = vectorize_text(df, 'normalize_supplier_name', max_features=100)  
    # Convert supplier name into numerical features with fewer dimensions.

    if 'Amount' in df.columns:  # Check if the dataset contains an 'Amount' column.
        scaler = StandardScaler()  # Initialize StandardScaler for normalization.
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])  
        # Normalize 'Amount' to bring it to a standard scale.
        
        numeric_features = df[['Amount_scaled']]  # Store numeric features in a DataFrame.
    else:
        numeric_features = pd.DataFrame()  # If 'Amount' is missing, create an empty DataFrame.

    features = pd.concat([gl_features, supplier_features, numeric_features], axis=1)  
    # Combine all extracted features into a single DataFrame.

    return features, gl_vectorizer, supplier_vectorizer  # Return the feature matrix and vectorizers.

        
if __name__ == "__main__":
    from preprocessing import load_data
    df_procurement, _ =load_data() 
    features, _, _ = prepare_features(df_procurement)
    print("features shape:",features.shape)

 



