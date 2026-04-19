import numpy as np
import scipy.io.wavfile as wavfile
import os

SAMPLE_RATE = 16000
def txt_to_wav(txt_file,wav_file):
    # Load the *.txt file to a list
    with open(txt_file,"r") as file:
        data = [int(line.strip()) for line in file if line.strip()]
    
    audio = np.array(data,dtype=np.int16)
    wavfile.write(wav_file,SAMPLE_RATE,audio)
    print(f"[INFO] converted {txt_file} to {wav_file}")

def make_dir(input_dir,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            txt_file = os.path.join(input_dir,filename)
            wav_file = os.path.join(output_dir,filename.replace(".txt",".wav"))
            txt_to_wav(txt_file,wav_file)

if __name__ == "__main__":
    input_folder = "data/raw/emergency"
    output_folder = "data/wav/emergency"
    make_dir(input_folder,output_folder)

    input_folder = "data/raw/assistance"
    output_folder = "data/wav/assistance"
    make_dir(input_folder,output_folder)

    input_folder = "data/raw/nourishment"
    output_folder = "data/wav/nourishment"
    make_dir(input_folder,output_folder)