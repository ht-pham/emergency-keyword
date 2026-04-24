import serial
import os
import argparse
import time

#-------------------------------------------
# Argument parser
#-------------------------------------------
parser = argparse.ArgumentParser(description="Audio Data Collector")
parser.add_argument("--port", type=str, required=True, help="Serial port to read from") # /dev/tty.usbmodemSN234567892
parser.add_argument("--label", type=str, required=True, help="Label name as assistance, emergency, or nourishment")
parser.add_argument("--samples", type=int, default=20, help="Number of recordings")
parser.add_argument("--duration",type=int, default=5, help="Duration of each recording")
args = parser.parse_args()

#-------------------------------------------
# Configure serial connection
#-------------------------------------------
BASE_DIR = "data/raw"
BAUD_RATE = 115200
SAMPLE_RATE = 16000
OUTPUT_DIR = os.path.join(BASE_DIR,args.label)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#-------------------------------------------
# Connect to the serial port
#-------------------------------------------
ser = serial.Serial(args.port,BAUD_RATE)
time.sleep(2.5)

print("[INFO] Waiting for Arduino to be ready...")
print("[INFO] Arduino ready!")
print(f"[INFO] Connected to {args.port} at {BAUD_RATE}")
#-------------------------------------------
# Collect audio samples
#-------------------------------------------
def record_samples(file_path, duration_sec):
    import numpy as np
    total_samples = SAMPLE_RATE * duration_sec
    collected = 0
    bytes_needed = total_samples * 2  # int16 = 2 bytes per sample
    print(f"[INFO] Recording to {file_path} for {duration_sec} seconds")
    time.sleep(0.5)
    try:
        with open(file_path, "w") as file:
            ser.reset_input_buffer()
            time.sleep(0.5)
            raw_data = b''

            while len(raw_data) < bytes_needed:
                chunk = ser.read(1024)  # read raw bytes
                raw_data += chunk
                print(f"DEBUG: Read {len(chunk)} bytes")

            # Convert bytes → int16 samples
            samples = np.frombuffer(raw_data[:bytes_needed], dtype=np.int16)

            # Write samples to txt (one per line)
            for s in samples:
                file.write(f"{int(s)}\n")
                collected += 1


            print(f"[INFO] File created: {file_path}")
            
    except KeyboardInterrupt:
        print("[INFO] Recording interrupted by user.")
    finally:
        print(f"[INFO] Saved {collected} samples to {file_path}")

#-------------------------------------------
# main loop to collect samples
#-------------------------------------------
existing_files = len(os.listdir(OUTPUT_DIR))

for i in range(args.samples):
    file_index = existing_files + i + 1
    filename = os.path.join(OUTPUT_DIR,f"{args.label}_{file_index:02d}.txt")

    input(f"[INFO] Press Enter to start recording sample {i+1}/{args.samples}...")
    record_samples(file_path=filename,duration_sec=args.duration)

print("[INFO] Data collection completed. Closing serial connection...")
ser.close()
