from mfcc_extractor import MFCCExtractor
from mfcc_extractor import wav_data_dir
import os

import numpy as np

def extract_mfcc_from_file(file_path):
    mfcc_extractor = MFCCExtractor()
    frames = mfcc_extractor.emphasize_and_frame(file_path)
    pow_frames = mfcc_extractor.ttf_conversion(frames)
    filter_banks = mfcc_extractor.mel_filter_bank(pow_frames)
    log_filter_banks = mfcc_extractor.logarithmic_scale(filter_banks)
    mfccs = mfcc_extractor.dct(log_filter_banks)
    
    return mfccs

def extract_mfcc_from_directory(directory):
    mfcc_features = []
    for file_name in os.listdir(wav_data_dir + directory):
        if file_name.endswith(".wav"):
            full_path = os.path.join(wav_data_dir + directory, file_name)
            mfcc_features.append(extract_mfcc_from_file(full_path))
            #mfcc_features[file_name] = extract_mfcc_from_file(full_path)
    return mfcc_features

def flatten_data(mfcc_features):
    X = []
    y = []

    for category,samples in mfcc_features.items():
        for sample in samples:
            X.append(sample)
            if category == 'nourishment':
                y.append(0)
            elif category == 'assistance':
                y.append(1)
            else:
                y.append(2)

    X = np.array(X)
    y = np.array(y)
    X = np.transpose(X,(0,2,1)) # changed from (180,298,12) to (180,12,298) for CNN model
    
    # Add channel to data for CNN and CNN-LSTM model
    #X = np.expand_dims(X,axis=1) # (180,1,12,298)

    return X,y

def export_mfcc_features(mfcc_features, output_file="./../../data/processed/features/mfcc_features.pkl"):
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(mfcc_features, f)


if __name__ == "__main__":
    categories = ["nourishment", "assistance", "emergency"]
    binary_categories = [0,1,2]

    # all_mfcc_features = {0:[60x[298,12]],1:[60x[298,12]],2:[60x[298,12]]}
    all_mfcc_features = {}
    for category in categories:
        all_mfcc_features[category] = extract_mfcc_from_directory(category)
        print(f"# MFCC features for {category}: {len(all_mfcc_features[category])}")

    X,y = flatten_data(all_mfcc_features)
    
    # Export to numpy files for later use
    np.save('./../../data/processed/labels/X.npy',X)
    np.save('./../../data/processed/labels/y.npy',y)
    
    export_mfcc_features(all_mfcc_features)