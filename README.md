### The Catholic University of America
#### CSC 549 - Final Project
# TinyML on Arduino: Emergency Keyword Detection System

### Abstract / Summary
The growing shortage of nursing staff in healthcare facilities has increased the need for assistive technologies. This project presents an improvement in patient monitoring and response time by using a TinyML-based speech recognition system that will be designed for emergency keyword detection on an embedded device. The device will monitor audio input and identify critical spoken cues such as “help,” “food,” “water”, and “bathroom”. This device will help patients with enabling hands-free  communication who perhaps are unable to access the traditional call systems. With this feature, audio data will be processed using Mel-frequency Cepstral Coefficients for extraction and classified using deep learning models that include Convolutional Neural Networks and Long Short-Term Memory networks. The trained models are optimized on a microcontroller for real-time inference. Upon the detection of the keywords that patients use, the system will trigger an alert to caregivers to be able to prioritize the patient's needs. This approach will demonstrate the real-time speech recognition on devices and highlight its potential to improve patient safety and healthcare efficiency.

### Methodology
#### Hardware
Arduino Nano 33 BLE Sense Rev2

#### Software Pipeline
1. Audio data collection: run recording scripts to record raw voice memos from Arduino device.

2. Feature extraction: run src/mfcc.py for feature extraction

3. Model training: train two different models on the same dataset for result evaluation.

4. TensorFlow Lite Conversion: convert the trained models to TinyML models

5. Model Deployment & Evaluation: upload the TFLite models into arduino/record_voices/record_voices.ino and run the sketch for inference and evaluation.

### How to run

1. Audio data collection:
From Arduino IDE, verify and upload arduino/record_voices/record_voices.ino to the board and keep IDE open for serial connection.

From your local IDE or your terminal at the root of project directory, for each label, run multiple times:

On MacOS run: 
```
python src/data/collect_audio.py --port /dev/cu.usbmodem[0-9] --label [label-name] --samples 10 --duration 10
```

Once you complete your recordings, you run this command line to convert *.txt to *.wav
```
python3 src/data/txt_to_wav.py 
```                                                                         

<!--python3 src/data/collect_audio.py --port /dev/cu.usbmodem11401 --label emergency --samples 10 --duration 3
python3 src/data/txt_to_wav.py-->

2. Feature Extraction:
