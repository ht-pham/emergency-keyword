import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cnn import CNN
from cnn_lstm import CNN_LSTM


def load_data(file_path="./../../data/processed/features/mfcc_features.csv"):
    # Load data from the CSV file
    df =pd.read_csv("./../../data/processed/features/mfcc_features.csv")
    
    # Merge all MFCC features into a single column and create labels
    df['mfcc_features'] = df.apply(lambda row: row['nourishment'] if row['nourishment']!="0" else (row['assistance'] if row['assistance'] != "0" else row['emergency']), axis=1)
    df['label'] = df.apply(lambda row: 0 if row['nourishment']!="0" else (1 if row['assistance'] != "0" else 2), axis=1)

    return df['mfcc_features'], df['label']

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    base_model = CNN()
    lstm_model = CNN_LSTM()

    

    
    



