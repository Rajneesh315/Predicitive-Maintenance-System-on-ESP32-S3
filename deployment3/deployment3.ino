/* Edge Impulse & ESP32-S3: Predictive Maintenance (Periodic Inference Only) */
#include <Rajneesh-project-1_inferencing.h>
#include <MPU6500_WE.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

/* ========== I2C PIN MAPPING ========== */
#define OLED_SDA 42
#define OLED_SCL 2
#define MPU_SDA 11
#define MPU_SCL 12

/* ========== OLED CONFIGURATION ========== */
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

/* ========== SENSOR CONFIGURATION ========== */
#define MPU_ADDR 0x68
MPU6500_WE myMPU = MPU6500_WE(&Wire1, MPU_ADDR);

// --- PREDICTIVE MAINTENANCE SETTINGS ---
const unsigned long CHECK_INTERVAL_MS = 4000; // Run inference every 10 seconds
unsigned long last_check_time = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    ei_printf("\n=== ESP32-S3 Predictive Maintenance ===\n");

    // 1. Initialize OLED on BUS 0
    Wire.begin(OLED_SDA, OLED_SCL);
    Wire.setClock(400000); 
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        ei_printf("FATAL: OLED allocation failed\n");
        while (1);
    }
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Booting System...");
    display.display();

    // 2. Initialize MPU6500 on BUS 1
    Wire1.begin(MPU_SDA, MPU_SCL);
    Wire1.setClock(100000); // 100kHz for stable I2C data
    if (!myMPU.init()) {
        ei_printf("FATAL: MPU6500 not found!\n");
        display.println("MPU6500 ERROR!");
        display.display();
        while (1);
    }
    
    // Set range (No autoOffsets to avoid boot calibration bugs)
    myMPU.setAccRange(MPU6500_ACC_RANGE_4G); 

    display.println("Sensors OK.");
    display.println("Waiting for interval...");
    display.display();
    ei_printf("System Ready. Running AI every %d seconds.\n", CHECK_INTERVAL_MS / 1000);
}

/**
 * @brief Records 1 second of data and runs the Edge Impulse Model
 */
void run_smart_inference() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(10, 20);
    display.println("SAMPLING...");
    display.display();

    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    // Fill buffer at exactly 100Hz
    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        uint64_t next_tick = micros() + (1000000 / EI_CLASSIFIER_FREQUENCY);

        xyzFloat accel = myMPU.getGValues();

        buffer[ix + 0] = accel.x;
        buffer[ix + 1] = accel.y;
        buffer[ix + 2] = accel.z;

        uint64_t time_to_wait = next_tick - micros();
        if (time_to_wait > 0) delayMicroseconds(time_to_wait);
    }

    display.clearDisplay();
    display.setCursor(10, 20);
    display.println("ANALYZING");
    display.display();

    // Convert to Edge Impulse signal
    signal_t features_signal;
    numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &features_signal);

    // Run Classifier
    ei_impulse_result_t result = { 0 };
    run_classifier(&features_signal, &result, false);

    // Extract values
    float score_bad = result.classification[0].value; 
    float score_normal = result.classification[1].value;
    float anomaly_score = result.anomaly;

    // Update OLED with Results
    display.clearDisplay();
    display.setTextSize(2);
    
    // Check if Anomaly triggers OR if the Neural Net detects the bad class
    // Threshold set to 1.5 to prevent overly strict K-Means alarms on normal operation
    if (anomaly_score > 7 || score_bad > 0.5) {
        display.setCursor(10, 5);
        display.println("ANOMALY!");
        display.setTextSize(1);
        display.setCursor(0, 30);
        display.print("Loose Bolt: ");
        display.print(score_bad * 100, 1);
        display.println("%");
    } else {
        display.setCursor(25, 5);
        display.println("NORMAL");
        display.setTextSize(1);
        display.setCursor(0, 30);
        display.print("Healthy: ");
        display.print(score_normal * 100, 1);
        display.println("%");
    }
    
    display.setCursor(0, 50);
    display.print("K-Means Anomaly: ");
    display.print(anomaly_score, 2);
    display.display();
}

void loop() {
    // Periodic Trigger: No gates, no checks. Just pure inference.
    if (millis() - last_check_time >= CHECK_INTERVAL_MS) {
        last_check_time = millis();
        ei_printf(">> Running Neural Network...\n");
        run_smart_inference();
    }
    
    delay(10); 
}