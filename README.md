# Predictive Maintenance System on ESP32-S3 using TinyML

![TinyML](https://img.shields.io/badge/TinyML-Edge_Impulse-blue)
![Platform](https://img.shields.io/badge/Platform-ESP32--S3-orange)
![Framework](https://img.shields.io/badge/Framework-Arduino_C++-teal)
![Status](https://img.shields.io/badge/Status-Completed-success)

A real-time, edge-AI predictive maintenance device built to detect mechanical anomalies (like loose bolts or failing bearings) in industrial fans/motors. This project uses an ESP32-S3 microcontroller to run a quantized Neural Network and a K-Means Anomaly Detector completely offline, identifying mechanical faults before catastrophic failure occurs.

## Project Overview
Industrial machines transition through subtle vibrational changes before failing. Traditional maintenance is either reactive (fixing broken parts) or calendar-based (wasting good parts). This system introduces **proactive maintenance** by analyzing real-time vibration data at the edge.

**Key Features:**
* **100Hz Real-Time Sampling:** Captures high-resolution vibrational data.
* **Dual-AI Architecture:** Uses a Neural Network for known fault classification and K-Means clustering for unknown anomaly detection.
* **Edge Processing:** Fully offline inference (~30ms total execution time) using Edge Impulse's EON Compiler.
* **Dual I2C Bus Design:** Isolates the high-speed OLED display from the highly sensitive accelerometer to prevent data corruption.

---

## Hardware Architecture

* **Microcontroller:** ESP32-S3 Dev Module
* **Sensor:** MPU6500 6-DOF Accelerometer & Gyroscope
* **Display:** 0.96" SSD1306 OLED Display (128x64)

### Dual I2C Bus Wiring
To ensure signal integrity, the system separates the OLED and the Sensor onto two distinct hardware I2C buses:

| Component | I2C Bus | SDA Pin | SCL Pin | Clock Speed |
| :--- | :--- | :--- | :--- | :--- |
| **SSD1306 OLED** | `Wire` (Bus 0) | GPIO 42 | GPIO 2 | 400 kHz |
| **MPU6500 Sensor** | `Wire1` (Bus 1) | GPIO 11 | GPIO 12 | 100 kHz |

*(Note: MPU6500 I2C speed was explicitly downclocked to 100kHz to prevent high-frequency signal degradation and `0xFF` byte corruption during 100Hz sampling).*

---

## Machine Learning Pipeline (Edge Impulse)

### 1. Data Collection & DSP
* **Window Size:** 1000ms (1 second of continuous data per inference).
* **Sampling Rate:** 100Hz (100 samples per axis).
* **Signal Processing:** Spectral Features block (FFT) extracts the Root Mean Square (RMS) energy and frequency peaks of the X, Y, and Z axes. A High-Pass filter is applied to strip out gravitational constants.

### 2. Neural Network (Classifier)
* **Architecture:** Fully connected dense network.
* **Classes:** `Normal Operation` vs. `Loose Bolt`.
* **Quantization:** Int8 (Optimized for MCU memory limits).
* **Accuracy:** **96.05%** on unseen holdout data.

### 3. K-Means Anomaly Detection
Acts as a fallback for *unknown* faults. If the motor experiences a failure state it was never trained on (e.g., a broken blade), the K-Means algorithm detects that the RMS energy falls outside the boundaries of the `Normal` clusters and flags it. 
* *Tuning Note:* Cluster count was optimized from K=8 to K=4 to prevent false positives from normal battery voltage fluctuations.

---

## Engineering Challenges Solved

Building hardware-level AI comes with unique physical constraints that were solved during development:

1. **I2C Signal Degradation (The "4.8G Glitch"):** Initially, the system registered impossible physical forces (>4.0 Gs of change in 10ms) while resting. This was diagnosed as I2C bus corruption due to running standard breadboard jumper wires at 400kHz. Dropping the MPU6500 bus (`Wire1`) to `100kHz` completely eliminated the glitching.
2. **Boot-Time Calibration Bugs:**
   Standard `autoOffsets()` commands assume the machine is perfectly still at boot. If the rig was bumped when plugged in, the zero-point was permanently skewed. This was solved by relying entirely on the AI's DSP High-Pass filter to mathematically ignore gravity, removing the need for fragile hardware calibration.
3. **The "Dual-Brain" Alarm:**
   Initially, the strict K-Means algorithm (K=8) threw anomaly alerts even when the Neural Network correctly classified the data as normal. The logic gate was refactored and the K-Means cluster count was relaxed (K=4), ensuring the two models worked in harmony.

---

## How to Run

1. Clone this repository.
2. Open `deployment2.ino` in the Arduino IDE.
3. Install the required libraries:
   * `Adafruit GFX` & `Adafruit SSD1306`
   * `MPU6500_WE`
4. Download your specific exported Edge Impulse `.zip` Arduino Library and install it via `Sketch > Include Library > Add .ZIP Library...`
5. Select **ESP32S3 Dev Module** as your board.
6. Compile and Upload.
7. Open the Serial Monitor at `115200` baud.

---

## Future Scope
As a continuing project, future iterations may include:
* **Custom PCB Design:** Moving from breadboard to a fabricated PCB for industrial durability.
* **Power Management:** Integrating a LiPo battery charging circuit and deep-sleep states for remote deployment.
* **IoT Dashboard:** Utilizing the ESP32-S3's native Wi-Fi capabilities to publish MQTT alerts to a cloud dashboard when an anomaly is detected.

---
*Developed by Rajneesh - USICT ECE*
