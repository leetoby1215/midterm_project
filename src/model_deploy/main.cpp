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
#define signalLength (4096)
#define note_limit (1024)

DA7212 audio;

Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);
DigitalOut led1(LED1);
InterruptIn btn1(SW2);
InterruptIn btn2(SW3);
EventQueue queue_uLCD_display(32 * EVENTS_EVENT_SIZE);
EventQueue queue_uLCD_control(32 * EVENTS_EVENT_SIZE);
EventQueue queue_audio(32 * EVENTS_EVENT_SIZE);
EventQueue queue_load_note(32 * EVENTS_EVENT_SIZE);
EventQueue queue_play_song(32 * EVENTS_EVENT_SIZE);
EventQueue queue_modesong_select(32 * EVENTS_EVENT_SIZE);
Thread thread_uLCD_display(osPriorityNormal);
Thread thread_uLCD_control(osPriorityNormal);
Thread thread_audio(osPriorityNormal);
Thread thread_load_note(osPriorityNormal);
Thread thread_play_song(osPriorityNormal);
Thread thread_modesong_select(osPriorityNormal);
Thread thread_DNN(osPriorityNormal, 4 * 1024);

int mode = 0;
int mode_tmp = mode;
int song = 0;
int song_tmp = song;
bool pause = true;
bool pause_tmp = pause;
bool is_select = false;
bool is_select_tmp = is_select;

float Signal[signalLength];
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
int serialCount = 0;

int note[4][12] {
    {65, 69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123},
    {131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247},
    {262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494},
    {523, 554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988}
};

int song_note[3][48];

void DNN();
void pause_switch();
void select_switch();
void uLCD_control();
void uLCD_display();
void modesong_select();
int PredictGesture(float* output);

void playNote(int freq);
void loadSignal();
void play_song();

int main(int argc, char* argv[]) {
    led1 = true;
    thread_uLCD_display.start(callback(&queue_uLCD_display, &EventQueue::dispatch_forever));
    thread_uLCD_control.start(callback(&queue_uLCD_control, &EventQueue::dispatch_forever));
    thread_audio.start(callback(&queue_audio, &EventQueue::dispatch_forever));
    thread_load_note.start(callback(&queue_load_note, &EventQueue::dispatch_forever));
    thread_play_song.start(callback(&queue_play_song, &EventQueue::dispatch_forever));
    thread_modesong_select.start(callback(&queue_modesong_select, &EventQueue::dispatch_forever));

    queue_uLCD_display.call(uLCD_display);
    queue_uLCD_control.call(uLCD_control);
    queue_load_note.call(loadSignal);
    queue_play_song.call(play_song);

    btn1.fall(queue_uLCD_display.event(select_switch));
    btn2.fall(queue_uLCD_display.event(pause_switch));

    thread_DNN.start(&DNN);

    while (true) {
        wait(1);
    }
}

void initial() {

}

void DNN() {
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
    bool should_clear_buffer = false;
    bool got_data = false;
    int gesture_index;
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }
    static tflite::MicroOpResolver<6> micro_op_resolver;
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::micro::Register_RESHAPE(), 1);
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D, tflite::ops::micro::Register_MAX_POOL_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;
    interpreter->AllocateTensors();

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
        got_data = ReadAccelerometer(error_reporter, model_input->data.f, input_length, should_clear_buffer);
        if (!got_data) {
            should_clear_buffer = false;
            continue;
        }
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
        }
        gesture_index = PredictGesture(interpreter->output(0)->data.f);
        should_clear_buffer = gesture_index < label_num;
        if (gesture_index < label_num) {
            error_reporter->Report(config.output_message[gesture_index]);
        }
    }
}

void uLCD_control() {
    while (true) {
        if (mode != mode_tmp || pause != pause_tmp || is_select != is_select_tmp || song != song_tmp)
            queue_uLCD_display.call(uLCD_display);
        mode_tmp = mode;
        pause_tmp = pause;
        is_select_tmp = is_select;
        song_tmp = song;
        wait(0.1);
    }
}

void select_switch() {
    if (pause)
        is_select = !is_select;
}

void pause_switch() {
    if (is_select)
        pause = !pause;
    if (pause) {
        queue_audio.call(playNote, 0);
    } else {
        for (int i = 0; i < 999; i++) {
            queue_audio.call(playNote, note[2][4]);
        }
        queue_audio.call(playNote, 0);
    }

}

void uLCD_display() {
    uLCD.cls();
    if (pause) {
        switch (mode) {
        case 1:
            if (is_select) {
                uLCD.printf("You select forward mode.\n");
            } else {
                uLCD.printf("Now is forward mode.\n");
            }
            break;
        case 2:
            if (is_select) {
                uLCD.printf("You select backward mode.\n");
            } else {
                uLCD.printf("Now is backward mode.\n");
            }
            break;
        case 3:
            if (is_select) {
                uLCD.printf("You select Taiko mode.\n");
            } else {
                uLCD.printf("Now is Taiko mode.\n");
            }
            break;
        default:
            if (is_select) {
                uLCD.printf("You select change songs mode.\n");
                switch (song) {
                case 0:
                    uLCD.printf("You select song0.\n");
                    break;
                case 1:
                    uLCD.printf("You select song1.\n");
                    break;
                case 2:
                    uLCD.printf("You select song2.\n");
                    break;
                case 3:
                    uLCD.printf("You select song3.\n");
                    break;
                default:
                    break;
                }
            } else {
                uLCD.printf("Now is change songs mode.\n");
            }
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

int PredictGesture(float* output) {
    static int continuous_count = 0;
    static int last_predict = -1;
    int this_predict = -1;
    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

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

    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }

    continuous_count = 0;
    last_predict = -1;
    return this_predict;
}

void playNote(int freq) {
    for (int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (Signal[(uint16_t) (i * freq * signalLength * 1.0 / kAudioSampleFrequency) % signalLength] * ((1<<16) - 1)) * 0.5;
    }
    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) {
        audio.spk.play(waveform, kAudioTxBufferSize);
    }
}

void loadSignal() {
    int i = 0;
    serialCount = 0;
    audio.spk.pause();
    while (i < signalLength) {
        if(pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if(serialCount == 7) {
                serialInBuffer[serialCount] = '\0';
                Signal[i] = (float) atof(serialInBuffer);
                serialCount = 0;
                i++;
            }
        }
    }
    queue_modesong_select.call(modesong_select);
}

void modesong_select() {
    char control;
    while (true) {
        if(pc.readable()) {
            control = pc.getc();
            if (!is_select) {
                if (control == 'w') {
                    if (pause) {
                        if (mode == 3) {
                            mode = 0;
                        } else {
                            mode++;
                        }
                    }
                }
                if (control == 's') {
                    if (pause) {
                        if (mode == 0) {
                            mode = 3;
                        } else {
                            mode--;
                        }
                    }
                }
            }
            if (is_select && mode == 0) {
                if (control == 'w') {
                    if (pause) {
                        if (song == 3) {
                            song = 0;
                        } else {
                            song++;
                        }
                    }
                }
                if (control == 's') {
                    if (pause) {
                        if (song == 0) {
                            song = 3;
                        } else {
                            song--;
                        }
                    }
                }
            }
        }
    }
}

void play_song() {
}