#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "note.h"
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

DA7212 audio;

uLCD_4DGL uLCD(D1, D0, D2);
Serial pc(USBTX, USBRX);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);

Timer debounce;
Timer Taiko;

EventQueue queue_sw3;
EventQueue queue_sw3_1;
EventQueue queue_sw3_2;
EventQueue queue_uLCD;
EventQueue queue_audio;
EventQueue queue_load_data;
EventQueue queue_special_effect;
EventQueue queue_Taiko_song;
EventQueue queue_fall_note;

Thread thread_DNN;
Thread thread_select_detect;
Thread thread_sw3;
Thread thread_sw3_1;
Thread thread_sw3_2;
Thread thread_uLCD;
Thread thread_audio;
Thread thread_load_data;
Thread thread_beat_detect;
Thread thread_special_effect;
Thread thread_Taiko_song;
Thread thread_fall_note;
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
/*****************************************************************************/
bool beat = false;
bool beat_tmp = beat;
char serialInBuffer[7];
int score = 0;
bool have_clean_the_beat = false;
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
/*********TAIKO*********/
void load_data();
void beat_detect();
void special_effect();
void Taiko_song_play();
void fall_note();

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
    thread_load_data.start(callback(&queue_load_data, &EventQueue::dispatch_forever));
    thread_special_effect.start(callback(&queue_special_effect, &EventQueue::dispatch_forever));
    thread_Taiko_song.start(callback(&queue_Taiko_song, &EventQueue::dispatch_forever));
    thread_fall_note.start(callback(&queue_fall_note, &EventQueue::dispatch_forever));
    thread_beat_detect.start(&beat_detect);
    queue_uLCD.call(uLCD_display);
    sw2.fall(&select_switch);
    sw3.fall(queue_sw3.event(pause_switch));
    while (true) {
        wait_ms(1000);
    }
}

void fall_note() {
    for (int j = 10; j <= 100; j += 10) {
        if (pause)
            break;
        if (have_clean_the_beat) {
            uLCD.line(50, j - 10, 78, j - 10, 0);
            break;
        }
        uLCD.line(50, j, 78, j, WHITE);
        wait_ms(0.1);
        uLCD.line(50, j - 10, 78, j - 10, 0);
    }
}

void Taiko_song_play() {
    for (int i = 0; i < Taiko_length; i++) {
        if (pause) {
            score = 0;
            break;
        }
        length = Taiko_noteLength[i];
        if (length == 0)
            break;
        have_clean_the_beat = false;
        if (Taiko_beatNote[i] == 1) {
            queue_fall_note.call(fall_note);
        }
        while(length > 0) {
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize * standard_note_length * 0.1 * length; ++j) {
                queue_audio.call(playNote, Taiko_song[i]);
            }
            if (length == 1) {
                queue_audio.call(playNote, 1);
            }
            length--;
            wait(standard_note_length);
        }
    }
    if (pause)
        return;
    Taiko.stop();
    Taiko.reset();
    uLCD.cls();
    uLCD.printf("Your score: %d\n", score);
    score = 0;
}

void special_effect() {
    if (length <= 4) {
        score++;
        have_clean_the_beat = true;
    }
}

void beat_detect() {
    while (true) {
        if (mode_index == 3 && is_select && !pause && Taiko.read() > 6) {
            if (beat_tmp != beat)
                queue_special_effect.call(special_effect);
            beat_tmp = beat;
        }
        wait_ms(100);
    }
}

void load_data() {
    int i = 0;
    int get_value;
    while(i < Taiko_length) {
        if (pc.readable()) {
            pc.gets(serialInBuffer, 8);
            get_value = (int) std::stoi(serialInBuffer);
            Taiko_noteLength[i] = (get_value / 10) % 100;
            Taiko_song[i] = get_value / 1000;
            Taiko_beatNote[i] = get_value % 10;
            if (Taiko_noteLength[i] == 0) {
                uLCD.printf("File load completed\n");
                uLCD.printf("You can push SW3 to start the game!\n");
                break;
            } else {
                i++;
            }
        }
    }
}

void playNote(int freq) {
    for (int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI / (double) (kAudioSampleFrequency / (double) freq)) * ((1<<16) - 1)) ;
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
            while (true) {
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
                if (pause)
                    break;
            }
            break;
        case 2:
            while (true) {
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
                if (pause)
                    break;
            }
            break;
        case 3:
            break;
        default:
            queue_audio.call(playNote, 1);
            break;
        }
    }
}

void uLCD_display() {
    uLCD.cls();
    uLCD.color(GREEN);
    if (pause) {
        switch (mode_index) {
        case 1:
            if (is_select) {
                uLCD.printf("You select forward mode.\n");
            } else {
                uLCD.printf("Change songs mode\n\n");
                uLCD.color(BLUE);
                uLCD.printf("Forward mode\n\n");
                uLCD.color(GREEN);
                uLCD.printf("Backward mode\n\n");
                uLCD.printf("Taiko mode\n\n");
            }
            break;
        case 2:
            if (is_select) {
                uLCD.printf("You select backward mode.\n");
            } else {
                uLCD.printf("Change songs mode\n\n");
                uLCD.printf("Forward mode\n\n");
                uLCD.color(BLUE);
                uLCD.printf("Backward mode\n\n");
                uLCD.color(GREEN);
                uLCD.printf("Taiko mode\n\n");
            }
            break;
        case 3:
            if (is_select) {
                uLCD.printf("You select Taiko mode.\n");
                uLCD.printf("Please load the data from python first.\n");
                queue_load_data.call(load_data);
            } else {
                uLCD.printf("Change songs mode\n\n");
                uLCD.printf("Forward mode\n\n");
                uLCD.printf("Backward mode\n\n");
                uLCD.color(BLUE);
                uLCD.printf("Taiko mode\n\n");
                uLCD.color(GREEN);
            }
            break;
        default:
            if (is_select) {
                uLCD.printf("You select change songs mode.\n");
                uLCD.printf("This song is:\n\"%s\"\n", song_name[song_index]);
                if (mode_index == 2) {
                    if (song_index == 0) {
                        uLCD.printf("The next song is:\n\"%s\"\n", song_name[song_number - 1]);
                    } else {
                        uLCD.printf("The next song is:\n\"%s\"\n", song_name[song_index - 1]);
                    }
                } else {
                    if (song_index == song_number - 1) {
                        uLCD.printf("The next song is:\n\"%s\"\n", song_name[0]);
                    } else {
                        uLCD.printf("The next song is:\n\"%s\"\n", song_name[song_index + 1]);
                    }
                }
            } else {
                uLCD.color(BLUE);
                uLCD.printf("Change songs mode\n\n");
                uLCD.color(GREEN);
                uLCD.printf("Forward mode\n\n");
                uLCD.printf("Backward mode\n\n");
                uLCD.printf("Taiko mode\n\n");
            }
            break;
        }
    } else {
        if (mode_index == 3) {
            Taiko.start();
            uLCD.text_width(6);
            uLCD.text_height(6);
            uLCD.color(RED);
            for (int i = 5; i >= 0; i--) {
                uLCD.locate(1, 1);
                uLCD.printf("%d", i);
                wait(1);
            }
            uLCD.cls();
            uLCD.line(10, 100, 118, 100, RED);
            queue_Taiko_song.call(Taiko_song_play);
        } else {
            uLCD.printf("You are playing the song!\n");
            uLCD.printf("This song is:\n\"%s\"\n", song_name[song_index]);
            if (song_index == song_number - 1) {
                uLCD.printf("The next song is:\n\"%s\"\n", song_name[0]);
            } else {
                uLCD.printf("The next song is:\n\"%s\"\n", song_name[song_index + 1]);
            }
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
            if (gesture_index == 1) {
                beat = !beat;
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