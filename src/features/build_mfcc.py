from mfcc import MFCCExtractor
from mfcc import wav_data_dir
import os

def extract_mfcc_from_file(file_path):
    mfcc_extractor = MFCCExtractor()
    frames = mfcc_extractor.emphasize_and_frame(file_path)
    pow_frames = mfcc_extractor.ttf_conversion(frames)
    filter_banks = mfcc_extractor.mel_filter_bank(pow_frames)
    log_filter_banks = mfcc_extractor.logarithmic_scale(filter_banks)
    mfccs = mfcc_extractor.dct(log_filter_banks)
    
    return mfccs

def extract_mfcc_from_directory(directory):
    mfcc_features = {}
    for file_name in os.listdir(wav_data_dir + directory):
        if file_name.endswith(".wav"):
            full_path = os.path.join(wav_data_dir + directory, file_name)
            mfcc_features[file_name] = extract_mfcc_from_file(full_path)
    return mfcc_features

def export_mfcc_features(mfcc_features, output_file="./../../data/processed/features/mfcc_features.pkl"):
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(mfcc_features, f)

if __name__ == "__main__":
    categories = ["nourishment", "assistance", "emergency"]
    binary_categories = [0,1,2]
    all_mfcc_features = {}
    for category in categories:
        all_mfcc_features[category] = extract_mfcc_from_directory(category)
        print(f"# MFCC features for {category}: {all_mfcc_features[category].keys()}")

    export_mfcc_features(all_mfcc_features)