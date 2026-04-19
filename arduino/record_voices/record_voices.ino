#include <PDM.h>

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 256

// default number of output channels
static const char channels = 1;

// default PCM output frequency
static const int frequency = 16000;

// Buffer to read samples into, each sample is 16-bits
short sampleBuffer[BUFFER_SIZE];

// Number of audio samples read
volatile int samplesRead;

// Buffer to accumulate samples before sending over serial
short transmitBuffer[4096];
int transmitBufferIndex = 0;


void onPDMdata(){
  // query the number of bytes available
  int bytesAvailable = PDM.available();

  // read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample
  samplesRead = bytesAvailable / 2;

  for (int i = 0; i < samplesRead; i++) {
    if (transmitBufferIndex < 4096) {
      transmitBuffer[transmitBufferIndex++] = sampleBuffer[i];
    }
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  while (!Serial) yield();

  // configure data receive callback
  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(BUFFER_SIZE);

  if (!PDM.begin(1,SAMPLE_RATE)){
    Serial.println("Failed to start PDM.");
    while (1) yield();
  }
  PDM.setGain(80);
  Serial.println("Recording...");
  
}

void loop() {
  // if (samplesRead) {
  //   // print samples
  //   for (int i=0;i < samplesRead; i++){
  //     Serial.println(sampleBuffer[i]);
  //   }
  //   // clear the read count
  //   samplesRead = 0;
  // }
  // Check if there are samples in the transmit buffer
  if (transmitBufferIndex > 0) {
    // Send the transmit buffer over the serial connection
    Serial.write((byte*)transmitBuffer, transmitBufferIndex * sizeof(short));

    // Clear the transmit buffer index
    transmitBufferIndex = 0;
  }
}


