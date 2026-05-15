/* Edge Impulse Arduino - Real-time Anomaly Detection for ESP32-S3
 * Optimized for MPU6500 with Triggered Sampling
 * 
 * This sketch reads accelerometer data from MPU6500, collects samples when triggered,
 * and runs anomaly detection inference in real-time.
 */

#include <Rajneesh-project-1_inferencing.h>
#include <MPU6500_WE.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

/* ========== PIN CONFIGURATION ========== */
#define MPU_SDA 11
#define MPU_SCL 12
#define MPU_ADDR 0x68

/* ========== DISPLAY CONFIGURATION ========== */
#define OLED_SDA 42
#define OLED_SCL 2
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

/* ========== SENSOR CONFIGURATION ========== */
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f

/* ========== TRIGGER CONFIGURATION ========== */
#define TRIGGER_PIN         10         // GPIO pin to trigger sampling (optional, or use Serial command)
#define TRIGGER_THRESHOLD   0.5f       // Accel threshold in G to auto-trigger sampling
#define USE_GPIO_TRIGGER    false      // Set true to use GPIO trigger, false for serial/auto-trigger
#define ANOMALY_THRESHOLD   0.5f       // Score above this = anomaly detected

/* ========== SAMPLING CONFIGURATION ========== */
#define SAMPLE_INTERVAL_MS  EI_CLASSIFIER_INTERVAL_MS  // From model config
#define SAMPLES_PER_CALL    3           // Accel has 3 axes (X, Y, Z)

/* ========== GLOBAL VARIABLES ========== */
MPU6500_WE myMPU(&Wire1, MPU_ADDR);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

static float sensor_data[3] = {0.0, 0.0, 0.0};
static bool sample_ready = false;
static bool trigger_sampling = false;
static unsigned long last_trigger_time = 0;
static const unsigned long DEBOUNCE_TIME = 500;  // ms between triggers

/**
 * @brief Initialize OLED Display on I2C Bus 0
 */
bool init_display(void) {
    Wire.begin(OLED_SDA, OLED_SCL);
    Wire.setClock(400000);  // 400kHz Fast Mode
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        ei_printf("ERROR: Failed to initialize OLED display!\n");
        return false;
    }
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Anomaly Detector");
    display.println("Initializing...");
    display.display();
    
    ei_printf("OLED display initialized successfully\n");
    return true;
}

/**
 * @brief Show startup screen
 */
void display_startup(void) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("=== ANOMALY DET ===");
    display.println("Model: " EI_CLASSIFIER_PROJECT_NAME);
    display.println("MPU6500: Ready");
    display.println("Display: Ready");
    display.println("");
    display.println("Waiting for trigger...");
    display.println("Send 's' to sample");
    display.display();
}

/**
 * @brief Show sampling in progress
 */
void display_sampling(int sample_count, int total_samples) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("SAMPLING");
    
    display.setTextSize(1);
    display.print("Progress: ");
    display.print(sample_count);
    display.print("/");
    display.println(total_samples);
    
    // Simple progress bar
    int bar_width = (sample_count * 120) / total_samples;
    display.drawRect(4, 30, 120, 8, SSD1306_WHITE);
    display.fillRect(4, 30, bar_width, 8, SSD1306_WHITE);
    
    display.display();
}

/**
 * @brief Show inference in progress
 */
void display_processing(void) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(20, 20);
    display.println("RUNNING");
    display.setCursor(15, 45);
    display.println("INFERENCE");
    display.display();
}

/**
 * @brief Show normal status (no anomaly)
 */
void display_normal(float anomaly_score) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(25, 5);
    display.println("NORMAL");
    
    display.setTextSize(1);
    display.setCursor(0, 30);
    display.print("Anomaly Score:");
    display.setCursor(0, 40);
    display.print(anomaly_score, 3);
    
    display.setTextSize(1);
    display.setCursor(0, 55);
    display.println("Status: OK");
    display.display();
}

/**
 * @brief Show anomaly detected
 */
void display_anomaly(float anomaly_score) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(10, 5);
    display.println("ANOMALY!");
    
    display.setTextSize(1);
    display.setCursor(0, 30);
    display.print("Score: ");
    display.println(anomaly_score, 3);
    
    display.setTextSize(2);
    display.setCursor(20, 42);
    display.println("WARNING!");
    display.display();
}

/**
 * @brief Initialize MPU6500 sensor on I2C Bus 1
 */
bool init_mpu6500(void) {
    Wire1.begin(MPU_SDA, MPU_SCL);
    Wire1.setClock(400000);  // 400kHz Fast Mode
    
    if (!myMPU.init()) {
        ei_printf("ERROR: Failed to initialize MPU6500!\n");
        return false;
    }
    
    ei_printf("MPU6500 initialized successfully\n");
    
    // Configure accelerometer
    myMPU.setAccRange(MPU6500_RANGE_4G);      // ±4G range
    myMPU.setAccDivider(1);                    // No additional filtering
    myMPU.enableAcc(true);
    
    ei_printf("Accelerometer configured: ±4G, output enabled\n");
    
    return true;
}

/**
 * @brief Read accelerometer data from MPU6500
 * Converts from G to m/s² and applies range clamping
 */
void read_mpu6500(void) {
    xyzFloat accel = myMPU.getAccRawValues();
    
    // Convert to G values
    sensor_data[0] = accel.x / 8192.0f;  // 4G range = 8192 LSB/G
    sensor_data[1] = accel.y / 8192.0f;
    sensor_data[2] = accel.z / 8192.0f;
    
    // Clamp to max range
    for (int i = 0; i < 3; i++) {
        if (fabs(sensor_data[i]) > MAX_ACCEPTED_RANGE) {
            sensor_data[i] = (sensor_data[i] >= 0) ? MAX_ACCEPTED_RANGE : -MAX_ACCEPTED_RANGE;
        }
        // Convert to m/s²
        sensor_data[i] *= CONVERT_G_TO_MS2;
    }
    
    sample_ready = true;
}

/**
 * @brief Check if motion exceeds threshold (for auto-trigger)
 */
bool check_motion_trigger(void) {
    float magnitude = sqrt(sensor_data[0]*sensor_data[0] + 
                          sensor_data[1]*sensor_data[1] + 
                          sensor_data[2]*sensor_data[2]);
    return (magnitude > (TRIGGER_THRESHOLD * CONVERT_G_TO_MS2));
}

/**
 * @brief Handle serial commands
 * 's' = start sampling
 * 'a' = toggle auto-trigger
 * '?' = print status
 */
void handle_serial_commands(void) {
    if (Serial.available()) {
        char cmd = Serial.read();
        
        switch (cmd) {
            case 's':
            case 'S':
                ei_printf(">> Sampling triggered via serial\n");
                trigger_sampling = true;
                break;
            case 'a':
            case 'A':
                // Toggle auto-trigger based on motion
                ei_printf(">> Auto-trigger toggled\n");
                break;
            case '?':
                ei_printf("Commands: 's'=sample, '?'=help\n");
                break;
            default:
                break;
        }
    }
}

/**
 * @brief Handle GPIO trigger (if enabled)
 */
void handle_gpio_trigger(void) {
    if (!USE_GPIO_TRIGGER) return;
    
    if (digitalRead(TRIGGER_PIN) == HIGH) {
        unsigned long now = millis();
        if (now - last_trigger_time > DEBOUNCE_TIME) {
            ei_printf(">> GPIO trigger detected\n");
            trigger_sampling = true;
            last_trigger_time = now;
        }
    }
}

/**
 * @brief Collect sensor samples for inference
 * Gathers enough samples to fill the model's input buffer
 */
bool collect_samples(void) {
    int total_samples = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / SAMPLES_PER_CALL;
    ei_printf("\nCollecting %d samples at %d ms intervals...\n", 
              total_samples, 
              SAMPLE_INTERVAL_MS);
    
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += SAMPLES_PER_CALL) {
        // Update display with progress
        display_sampling((ix / SAMPLES_PER_CALL) + 1, total_samples);
        
        // Calculate next sample time
        int64_t next_tick = (int64_t)micros() + ((int64_t)SAMPLE_INTERVAL_MS * 1000);
        
        // Read sensor
        read_mpu6500();
        
        // Copy to buffer
        memcpy(&buffer[ix], sensor_data, SAMPLES_PER_CALL * sizeof(float));
        
        ei_printf("Sample %d: X=%.2f, Y=%.2f, Z=%.2f m/s²\n",
                  ix / SAMPLES_PER_CALL, sensor_data[0], sensor_data[1], sensor_data[2]);
        
        // Wait for next sample time
        int64_t wait_time = next_tick - (int64_t)micros();
        if (wait_time > 0) {
            delayMicroseconds(wait_time);
        }
    }
    
    // Show processing screen
    display_processing();
    delay(500);
    
    // Create signal from buffer
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("ERROR: Failed to create signal from buffer (err: %d)\n", err);
        return false;
    }
    
    // Run inference
    ei_impulse_result_t result = {0};
    err = run_classifier(&signal, &result, false);
    
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERROR: Failed to run classifier (err: %d)\n", err);
        return false;
    }
    
    // Print results
    ei_printf("\n--- INFERENCE RESULTS ---\n");
    ei_printf("Timing - DSP: %d ms, Classification: %d ms, Anomaly: %d ms\n",
              result.timing.dsp, result.timing.classification, result.timing.anomaly);
    
    // Print classifications (if model has classes)
    if (EI_CLASSIFIER_LABEL_COUNT > 0) {
        ei_printf("Classifications:\n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("  %s: %.5f\n", 
                      result.classification[ix].label, 
                      result.classification[ix].value);
        }
    }
    
    // Print anomaly score and update display
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly Score: %.3f", result.anomaly);
    if (result.anomaly > ANOMALY_THRESHOLD) {
        ei_printf(" ⚠️  ANOMALY DETECTED!\n");
        display_anomaly(result.anomaly);
    } else {
        ei_printf(" ✓ Normal\n");
        display_normal(result.anomaly);
    }
    #else
    // If no anomaly support, just show results
    display_normal(0.0f);
    #endif
    
    delay(3000);  // Show result for 3 seconds
    
    return true;
}

/**
 * @brief Arduino setup
 */
void setup(void) {
    Serial.begin(115200);
    while (!Serial);  // Wait for USB connection
    
    ei_printf("\n=== ESP32-S3 Anomaly Detection System ===\n");
    ei_printf("Using MPU6500 on I2C Bus 1 (SDA=%d, SCL=%d)\n", MPU_SDA, MPU_SCL);
    ei_printf("Using OLED on I2C Bus 0 (SDA=%d, SCL=%d)\n", OLED_SDA, OLED_SCL);
    ei_printf("Model: %s\n", EI_CLASSIFIER_PROJECT_NAME);
    ei_printf("Input: %d samples @ %d ms interval\n", 
              EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / SAMPLES_PER_CALL,
              SAMPLE_INTERVAL_MS);
    
    // Initialize display first (Bus 0)
    if (!init_display()) {
        ei_printf("FATAL: Display initialization failed!\n");
        while(1);  // Halt
    }
    
    // Initialize sensor (Bus 1)
    if (!init_mpu6500()) {
        ei_printf("FATAL: Sensor initialization failed!\n");
        display.clearDisplay();
        display.setTextSize(1);
        display.setCursor(0, 0);
        display.println("MPU6500 ERROR!");
        display.display();
        while(1);  // Halt
    }
    
    // Configure GPIO trigger (if enabled)
    if (USE_GPIO_TRIGGER) {
        pinMode(TRIGGER_PIN, INPUT_PULLDOWN);
        ei_printf("GPIO Trigger enabled on pin %d\n", TRIGGER_PIN);
    }
    
    // Show startup screen
    display_startup();
    
    ei_printf("\nREADY - Waiting for trigger...\n");
    ei_printf("Send 's' via Serial to start sampling\n");
}

/**
 * @brief Arduino main loop
 * Monitors for triggers and runs inference when needed
 */
void loop(void) {
    // Check for serial commands
    handle_serial_commands();
    
    // Check for GPIO trigger
    handle_gpio_trigger();
    
    // Check for motion-based auto-trigger (optional)
    read_mpu6500();
    if (check_motion_trigger()) {
        unsigned long now = millis();
        if (now - last_trigger_time > DEBOUNCE_TIME) {
            ei_printf(">> Motion trigger (magnitude exceeded threshold)\n");
            trigger_sampling = true;
            last_trigger_time = now;
        }
    }
    
    // If triggered, collect samples and run inference
    if (trigger_sampling) {
        trigger_sampling = false;
        collect_samples();
        display_startup();  // Return to ready state
        ei_printf("\nREADY - Waiting for next trigger...\n");
    }
    
    // Small delay to prevent overwhelming the loop
    delay(10);
}
