#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text feature engineering for clinical trial data

This script applies NLP techniques to extract features from text fields in clinical trial data,
such as eligibility criteria and intervention descriptions.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
def download_nltk_resources():
    """
    Download required NLTK resources if not already available
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'

class TextPreprocessor:
    """
    Class for preprocessing text data from clinical trials
    """
    def __init__(self):
        download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean and normalize text data
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        
        Args:
            text: Cleaned text string
            
        Returns:
            list: List of lemmatized tokens
        """
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        
        return lemmatized
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        
        Args:
            text: Raw text string
            
        Returns:
            str: Preprocessed text ready for feature extraction
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)

class TextFeatureExtractor:
    """
    Class for extracting features from text data
    """
    def __init__(self, max_features=1000, n_components=50):
        self.preprocessor = TextPreprocessor()
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    def extract_medical_terms(self, text):
        """
        Extract medical terms and concepts from text
        
        Args:
            text: Raw text string
            
        Returns:
            dict: Dictionary of extracted medical features
        """
        if not isinstance(text, str):
            return {}
        
        features = {}
        
        # Extract inclusion/exclusion criteria indicators
        features['has_inclusion'] = 1 if re.search(r'inclusion|include', text, re.I) else 0
        features['has_exclusion'] = 1 if re.search(r'exclusion|exclude', text, re.I) else 0
        
        # Extract age restrictions
        features['mentions_age'] = 1 if re.search(r'\bage\b|years old', text, re.I) else 0
        
        # Extract common cancer treatments
        features['mentions_chemotherapy'] = 1 if re.search(r'chemotherapy|chemo', text, re.I) else 0
        features['mentions_radiation'] = 1 if re.search(r'radiation|radiotherapy', text, re.I) else 0
        features['mentions_surgery'] = 1 if re.search(r'surgery|surgical|resection', text, re.I) else 0
        features['mentions_immunotherapy'] = 1 if re.search(r'immunotherapy|immune', text, re.I) else 0
        
        # Extract common cancer types
        features['mentions_breast_cancer'] = 1 if re.search(r'breast cancer', text, re.I) else 0
        features['mentions_lung_cancer'] = 1 if re.search(r'lung cancer', text, re.I) else 0
        features['mentions_prostate_cancer'] = 1 if re.search(r'prostate cancer', text, re.I) else 0
        features['mentions_colorectal_cancer'] = 1 if re.search(r'colorectal|colon cancer', text, re.I) else 0
        
        # Extract biomarker mentions
        features['mentions_biomarker'] = 1 if re.search(r'biomarker|marker|mutation|expression', text, re.I) else 0
        
        # Extract comorbidity mentions
        features['mentions_comorbidity'] = 1 if re.search(r'comorbid|comorbidity|condition', text, re.I) else 0
        
        return features
    
    def fit_transform_tfidf_svd(self, texts):
        """
        Fit and transform text data using TF-IDF and SVD
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy.ndarray: Matrix of text features
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Fit and transform with TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_texts)
        
        # Apply dimensionality reduction with SVD
        svd_matrix = self.svd.fit_transform(tfidf_matrix)
        
        return svd_matrix
    
    def transform_tfidf_svd(self, texts):
        """
        Transform new text data using fitted TF-IDF and SVD
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy.ndarray: Matrix of text features
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Transform with TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(preprocessed_texts)
        
        # Apply dimensionality reduction with SVD
        svd_matrix = self.svd.transform(tfidf_matrix)
        
        return svd_matrix
    
    def get_feature_names(self):
        """
        Get feature names for the SVD components
        
        Returns:
            list: List of feature names
        """
        return [f'svd_component_{i}' for i in range(self.n_components)]

def extract_text_features(df, text_column, prefix=''):
    """
    Extract features from a text column in the DataFrame
    
    Args:
        df: DataFrame containing the data
        text_column: Name of the column containing text data
        prefix: Prefix for the feature column names
        
    Returns:
        pandas.DataFrame: DataFrame with extracted text features
    """
    if text_column not in df.columns:
        print(f"Column {text_column} not found in DataFrame")
        return df
    
    # Initialize feature extractor
    extractor = TextFeatureExtractor(max_features=1000, n_components=20)
    
    # Extract medical terms
    print(f"Extracting medical terms from {text_column}...")
    medical_features = df[text_column].apply(extractor.extract_medical_terms)
    
    # Convert list of dictionaries to DataFrame
    medical_df = pd.DataFrame(medical_features.tolist())
    
    # Add prefix to column names
    if prefix:
        medical_df = medical_df.add_prefix(f"{prefix}_")
    
    # Extract TF-IDF and SVD features
    print(f"Extracting TF-IDF and SVD features from {text_column}...")
    texts = df[text_column].fillna("").tolist()
    svd_matrix = extractor.fit_transform_tfidf_svd(texts)
    
    # Create DataFrame with SVD features
    feature_names = extractor.get_feature_names()
    if prefix:
        feature_names = [f"{prefix}_{name}" for name in feature_names]
    
    svd_df = pd.DataFrame(svd_matrix, columns=feature_names, index=df.index)
    
    # Combine all features
    result_df = pd.concat([df, medical_df, svd_df], axis=1)
    
    print(f"Added {len(medical_df.columns) + len(svd_df.columns)} text features")
    return result_df

def process_all_text_fields(df):
    """
    Process all relevant text fields in the clinical trial data
    
    Args:
        df: DataFrame containing the clinical trial data
        
    Returns:
        pandas.DataFrame: DataFrame with all text features added
    """
    # Process eligibility criteria
    if 'EligibilityCriteria' in df.columns:
        df = extract_text_features(df, 'EligibilityCriteria', prefix='elig')
    
    # Process intervention descriptions
    if 'InterventionName' in df.columns:
        df = extract_text_features(df, 'InterventionName', prefix='interv')
    
    # Process official title
    if 'OfficialTitle' in df.columns:
        df = extract_text_features(df, 'OfficialTitle', prefix='title')
    
    return df

def main():
    """
    Main function to execute the text feature engineering pipeline
    """
    try:
        # Find the most recent processed data file
        csv_files = list(PROCESSED_DATA_DIR.glob("processed_oncology_trials_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No processed data files found")
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        input_file = csv_files[0]
        
        print(f"Loading processed data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Process all text fields
        df_with_text_features = process_all_text_fields(df)
        
        # Save the enhanced dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROCESSED_DATA_DIR / f"oncology_trials_with_text_features_{timestamp}.csv"
        df_with_text_features.to_csv(output_file, index=False)
        
        print(f"Saved data with text features to {output_file}")
        print(f"Total features: {len(df_with_text_features.columns)}")
        
        return df_with_text_features
    
    except Exception as e:
        print(f"Error in text feature engineering pipeline: {e}")
        return None

if __name__ == "__main__":
    from datetime import datetime  # Import here to avoid circular import
    main()