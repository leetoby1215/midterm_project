#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "uLCD_4DGL.h"
#include "DA7212.h"
#include <cmath>

#define bufferLength (32)
#define note_limit (1024)

DA7212 audio;
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);
DigitalOut green_led(LED2);
InterruptIn btn1(SW2);
InterruptIn btn2(SW3);
EventQueue queue_uLCD(32 * EVENTS_EVENT_SIZE);
EventQueue queue_audio(32 * EVENTS_EVENT_SIZE);
EventQueue queue_load_note(32 * EVENTS_EVENT_SIZE);
Thread thread_uLCD;
Thread thread_audio;
Thread thread_load_note;

int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
int serialCount = 0;

int mode = 0;
int mode_tmp = mode;
bool pause = true;
bool pause_tmp = pause;

float freqency;
float duration;

int note[4][12] {
    {65, 69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123},
    {131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247},
    {262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494},
    {523, 554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988}
};

char song_name[10][18];
int song_lengh[10];
int song[2][note_limit];

void mode_change();
void pause_switch();
void uLCD_display();
int PredictGesture(float* output);

void playNote();

void Node_set(float freq, float dura);
void load_note();

int main(int argc, char* argv[]) {
    
    thread_uLCD.start(callback(&queue_uLCD, &EventQueue::dispatch_forever));
    thread_audio.start(callback(&queue_audio, &EventQueue::dispatch_forever));
    thread_load_note.start(callback(&queue_load_note, &EventQueue::dispatch_forever));

    queue_uLCD.call(uLCD_display);
    queue_audio.call(playNote);
    queue_load_note.call(load_note);

    btn1.fall(queue_uLCD.event(mode_change));
    btn2.fall(queue_uLCD.event(pause_switch));
    //queue_audio(queue_audio.event(stopPlayNoteC));
    while (true) {
        if (mode != mode_tmp || pause != pause_tmp)
            queue_uLCD.call(uLCD_display);
        mode_tmp = mode;
        pause_tmp = pause;
        wait(0.1);
    }
    /*// Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;
    // The gesture index of the prediction
    int gesture_index;

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::micro::Register_RESHAPE(), 1);
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D, tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());

    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor
    TfLiteTensor* model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }
    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
    }
    error_reporter->Report("Set up successful...\n");

    while (true) {
        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f, input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
            should_clear_buffer = false;
            continue;
        }
        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);
        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        if (gesture_index < label_num) {
            error_reporter->Report(config.output_message[gesture_index]);
        }
        wait(1);
    }*/
}

void mode_change() {
    if (pause) {
        if (mode == 3) {
            mode = 0;
        } else {
            mode++;
        }
    }
}

void pause_switch() {
    pause = !pause;
}

void uLCD_display() {
    uLCD.cls();
    if (pause) {
        switch (mode) {
        case 1: uLCD.printf("Now is forward mode.\n");
            break;
        case 2: uLCD.printf("Now is backward mode.\n");
            break;
        case 3: uLCD.printf("Now is Taiko mode.\n");
            break;
        default: uLCD.printf("Now is change songs mode.\n");
            break;
        }
    } else {
        if (mode == 3) {
            uLCD.printf("You are playing Taiko!");
        } else {
            uLCD.printf("You are playing the song!");
        }
    }
}

// Return the result of the last prediction
int PredictGesture(float* output) {
    // How many times the most recent gesture has been matched in a row
    static int continuous_count = 0;
    // The result of the last prediction
    static int last_predict = -1;
    // Find whichever output has a probability > 0.8 (they sum to 1)
    int this_predict = -1;
    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

    // No gesture was detected above the threshold
    if (this_predict == -1) {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }
    if (last_predict == this_predict) {
        continuous_count += 1;
    } else {
        continuous_count = 0;
    }
    last_predict = this_predict;

    // If we haven't yet had enough consecutive matches for this gesture,
    // report a negative result
    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }
    // Otherwise, we've seen a positive result, so clear all our variables
    // and report it
    continuous_count = 0;
    last_predict = -1;
    return this_predict;
}

void playNote() {
    while (true) {
        for (int i = 0; i < duration * 1000; i++) {
            for (int i = 0; i < kAudioTxBufferSize; i++) {
                waveform[i] = (int16_t) (sin((double)i * 2. * M_PI / (double) (kAudioSampleFrequency / freqency)) * ((1<<16) - 1)) * 0.5;
            }
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) {
                audio.spk.play(waveform, kAudioTxBufferSize);
            }
        }
    }
}

void Node_set(float freq, float dura) {
    freqency = freq;
    duration = 1.0 / dura;
    wait(1.0 / dura);
    freqency = 0;
}

void load_note() {
    int i = 0;
    int j = 0;
    int note_counter = 0;
    char serialInBuffer[5];
    char i_in[1];
    char j_in[2];
    char dura_in[2];
    serialCount = 0;
    while(true) {
        if(pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if(serialCount == 5) {
                i_in[0] = serialInBuffer[0];
                j_in[0] = serialInBuffer[1];
                j_in[1] = serialInBuffer[2];
                dura_in[0] = serialInBuffer[3];
                dura_in[1] = serialInBuffer[4];
                i = (int) atoi(i_in);
                j = (int) atoi(j_in);
                Node_set(note[i][j], atoi(dura_in));
                serialCount = 0;
                note_counter++;
            }
        }
    }
}