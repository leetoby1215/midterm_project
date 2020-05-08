#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "uLCD_4DGL.h"
#include "DA7212.h"
#include <cmath>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define standard_note_length 0.1
#define song_number 3
#define song_length 500

DA7212 audio;

uLCD_4DGL uLCD(D1, D0, D2);
Serial pc(USBTX, USBRX);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);

Timer debounce;

EventQueue queue_sw3;
EventQueue queue_sw3_1;
EventQueue queue_sw3_2;
EventQueue queue_uLCD;
EventQueue queue_audio;

Thread thread_DNN;
Thread thread_select_detect;
Thread thread_sw3;
Thread thread_sw3_1;
Thread thread_sw3_2;
Thread thread_uLCD;
Thread thread_audio;

/*****************************************************************************/
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
bool should_clear_buffer = false;
bool got_data = false;
int gesture_index;
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
static tflite::MicroOpResolver<6> micro_op_resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* model_input;
int input_length;
/*****************************************************************************/
int mode_index = 0;
int mode_index_tmp = mode_index;
int song_index = 0;
int song_index_tmp = song_index;
bool is_select = false;
bool is_select_tmp = is_select;
bool pause = true;
bool pause_tmp = pause;
/*****************************************************************************/
int16_t waveform[kAudioTxBufferSize];
int length;
char song_name[song_number][18] = {
    "moonlight_0",
    "moonlight_1",
    "moonlight_2"
};

int note[6][12] {
    33, 35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62,
    65, 69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123,
    131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247,
    262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494,
    523, 554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988,
    1047, 1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976
};

int song[song_number][song_length] = {
    {
    // 1
    note[1][1], note[1][8], note[2][1], note[2][4], note[2][8], note[2][1], note[2][4], note[2][8],
    note[3][1], note[2][4], note[2][8], note[3][1], note[3][4], note[2][8], note[3][1], note[3][4],
    note[3][8], note[3][1], note[3][4], note[3][8], note[4][1], note[3][4], note[3][8], note[4][1],
    note[4][4], note[3][8], note[4][1], note[4][4], note[4][8], note[4][8],
    // 3
    note[1][0], note[1][8], note[2][0], note[2][3], note[2][8], note[2][0], note[2][3], note[2][8],
    note[3][0], note[2][3], note[2][8], note[3][0], note[3][3], note[2][8], note[3][0], note[3][3],
    note[3][8], note[3][0], note[3][3], note[3][8], note[4][0], note[3][3], note[3][8], note[4][0],
    note[4][3], note[3][8], note[4][0], note[4][3], note[4][8], note[4][8],
    // 5
    note[0][11], note[2][1], note[2][5], note[2][8], note[3][1], note[2][5], note[2][8], note[3][1],
    note[3][5], note[2][8], note[3][1], note[3][5], note[2][8], note[3][1], note[3][5], note[3][8],
    note[4][1], note[3][5], note[3][8], note[4][1], note[4][5], note[3][8], note[4][1], note[4][5],
    note[4][8], note[4][1], note[4][5], note[4][8], note[5][1], note[5][1],
    // 7
    note[0][8], note[1][1], note[1][6], note[1][9], note[3][1], note[3][1], note[3][6], note[3][8],
    note[4][1], note[4][1], note[4][6], note[4][8], note[5][1], note[5][1],
    note[0][8], note[1][1], note[1][4], note[1][7], note[3][1], note[3][1], note[3][4], note[3][7],
    note[4][1], note[4][1], note[4][4], note[4][7], note[5][1], note[5][1],
    // 9
    note[5][0], note[3][8], note[4][8], note[3][8], note[4][8], note[3][10], note[4][8],
    note[4][0], note[4][8], note[5][1], note[4][8], note[4][3], note[4][8], note[4][0], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][6], note[4][8], note[4][4], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][0], note[4][8], note[3][9], note[4][7],
    // 11
    note[3][8], note[5][0], note[3][8], note[4][8], note[3][8], note[4][8], note[3][10], note[4][8],
    note[4][0], note[4][8], note[5][1], note[4][8], note[4][3], note[4][8], note[4][0], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][6], note[4][8], note[4][4], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][0], note[4][8], note[3][9], note[4][7],
    // 13
    note[3][8], note[4][8], note[3][9], note[4][7], note[3][8], note[4][8], note[3][9], note[4][7],
    note[3][8], note[4][8], note[3][9], note[4][7], note[3][8], note[4][8], note[3][9], note[4][7],
    note[3][8], note[2][8],
    // 15
    note[1][1], note[1][8], note[2][1], note[2][4], note[2][8], note[2][1], note[2][4], note[2][8],
    note[3][1], note[2][4], note[2][8], note[3][1], note[3][4], note[2][8], note[3][1], note[3][4],
    note[3][8], note[3][1], note[3][4], note[3][8], note[4][1], note[3][4], note[3][8], note[4][1],
    note[4][4], note[4][4], note[4][8], note[5][1], note[5][4], note[5][4],
    // 17
    note[0][10], note[1][4], note[1][7], note[3][1], note[3][4], note[1][7], note[3][1], note[3][4],
    note[3][7], note[3][1], note[3][4], note[3][7], note[4][1], note[3][4], note[3][7], note[4][1],
    note[4][4], note[3][7], note[4][1], note[4][4], note[4][7], note[4][1], note[4][4], note[4][7],
    note[5][1], note[4][4], note[4][7], note[5][1], note[5][4], note[5][4],
    // 19
    note[0][7], note[1][3], note[2][10], note[3][1], note[3][3], note[2][10], note[3][1], note[3][3],
    note[3][10], note[3][1], note[3][3], note[3][10], note[4][1], note[3][3], note[3][10], note[4][1],
    note[4][3], note[3][10], note[4][1], note[4][3], note[4][10], note[4][1], note[4][3], note[4][10],
    note[5][1], note[4][10], note[4][3], note[4][1], note[4][10], note[4][3], note[4][1], note[3][10]
    }, {0}, {0} // note[][], note[][], note[][], note[][], note[][], note[][], note[][], note[][],
};

int noteLength[song_number][song_length] = {
    {
    // 1
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 3
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 5
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 7
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 9
    2, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 11
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 13
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    4, 12,
    // 15
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 17
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 19
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {0}, {0}
};
/*****************************************************************************/

/*********DNN FUNCTIONS*********/
void initial();
void DNN();
int PredictGesture(float* output);
/*********SELECT FUNCTIONS*********/
void select_detect();
void select_switch();
void pause_switch();
void pause_switch_1();
void pause_switch_2();
/*********uLCD PISPLAY FINCTIONS*********/
void uLCD_display(); 
/*********AUDIO FUNCTIONS*********/
void playNote(int freq);

int main(int argc, char* argv[]) {
    initial();
    debounce.start();   
    thread_DNN.start(&DNN);
    thread_select_detect.start(&select_detect);
    thread_sw3.start(callback(&queue_sw3, &EventQueue::dispatch_forever));
    thread_sw3_1.start(callback(&queue_sw3_1, &EventQueue::dispatch_forever));
    thread_sw3_2.start(callback(&queue_sw3_2, &EventQueue::dispatch_forever));
    thread_uLCD.start(callback(&queue_uLCD, &EventQueue::dispatch_forever));
    thread_audio.start(callback(&queue_audio, &EventQueue::dispatch_forever));
    queue_uLCD.call(uLCD_display);
    sw2.fall(&select_switch);
    sw3.fall(queue_sw3.event(pause_switch));
    while (true) {
        wait(1);
    }
}

void song_function_0() {
    for(int i = 0; i < song_length; i++) {
        if (pause)
            break;
        length = noteLength[i];
        if (length == 0)
            break;
        while(length > 0) {
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize * standard_note_length * 0.1 * length; ++j) {
                queue_audio.call(playNote, song[i]);
            }
            if (length == 1) {
                queue_audio.call(playNote, 1);
            }
            length--;
            wait(standard_note_length);
        }
    }
}

void playNote(int freq) {
    for (int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI / (double) (kAudioSampleFrequency / (double) freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
}

void select_switch() {
    if (debounce.read_ms() > 500) {
        if (pause)
            is_select = !is_select;
        debounce.reset();
    }
}

void pause_switch() {
    if (debounce.read_ms() > 500) {
        queue_sw3_1.call(pause_switch_1);
        queue_sw3_2.call(pause_switch_2);
        debounce.reset();
    }
}

void pause_switch_1() {
    if (is_select)
        pause = !pause;
}

void pause_switch_2() {
    if (!pause) {
        switch (mode_index) {
        case 0:
            for(int i = 0; i < song_length; i++) {
                if (pause)
                    break;
                length = noteLength[song_index][i];
                if (length == 0)
                    break;
                while(length > 0) {
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize * standard_note_length * 0.1 * length; ++j) {
                        queue_audio.call(playNote, song[song_index][i]);
                    }
                    if (length == 1) {
                        queue_audio.call(playNote, 1);
                    }
                    length--;
                    wait(standard_note_length);
                }
            }
            pause = true;
            break;
        case 1:
            for (song_index = 0; song_index < song_number; song_index++) {
                for(int i = 0; i < song_length; i++) {
                    if (pause)
                        break;
                    length = noteLength[song_index][i];
                    if (length == 0)
                        break;
                    while(length > 0) {
                        for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize * standard_note_length * 0.1 * length; ++j) {
                            queue_audio.call(playNote, song[song_index][i]);
                        }
                        if (length == 1) {
                            queue_audio.call(playNote, 1);
                        }
                        length--;
                        wait(standard_note_length);
                    }
                }
            }
            pause = true;
            break;
        case 2:
            for (song_index = song_number - 1; song_index >= 0; song_index--) {
                for(int i = 0; i < song_length; i++) {
                    if (pause)
                        break;
                    length = noteLength[song_index][i];
                    if (length == 0)
                        break;
                    while(length > 0) {
                        for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize * standard_note_length * 0.1 * length; ++j) {
                            queue_audio.call(playNote, song[song_index][i]);
                        }
                        if (length == 1) {
                            queue_audio.call(playNote, 1);
                        }
                        length--;
                        wait(standard_note_length);
                    }
                }
            }
            pause = true;
            break;
        default:
            queue_audio.call(playNote, 1);
            break;
        }
    }
}

void uLCD_display() {
    uLCD.cls();
    if (pause) {
        switch (mode_index) {
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
                uLCD.printf("This song is:\n\"%s\"\n", song_name[song_index]);
            } else {
                uLCD.printf("Now is change songs mode.\n");
            }
            break;
        }
    } else {
        if (mode_index == 3) {
            uLCD.printf("You are playing Taiko!\n");
        } else {
            uLCD.printf("You are playing the song!\n");
            uLCD.printf("This song is:\n\"%s\"\n", song_name[song_index]);
        }
    }
}

void select_detect() {
    while (true) {
        if (mode_index != mode_index_tmp || song_index != song_index_tmp || is_select != is_select_tmp || pause != pause_tmp)
            queue_uLCD.call(uLCD_display);
        mode_index_tmp = mode_index;
        song_index_tmp = song_index;
        is_select_tmp = is_select;
        pause_tmp = pause;
        wait_ms(100);
    }
}

void initial() {
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::micro::Register_RESHAPE(), 1);
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D, tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }
    input_length = model_input->bytes / sizeof(float);
    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
    }
    error_reporter->Report("Set up successful...\n");
}

void DNN() {
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
            if (!is_select) {
                if (gesture_index == 0) {
                    if (pause) {
                        if (mode_index == 3) {
                            mode_index = 0;
                        } else {
                            mode_index++;
                        }
                    }
                }
            }
            if (is_select && mode_index == 0) {
                if (gesture_index == 0) {
                    if (pause) {
                        if (song_index == song_number - 1) {
                            song_index = 0;
                        } else {
                            song_index++;
                        }
                    }
                }
            }
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