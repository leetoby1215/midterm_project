#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "uLCD_4DGL.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

uLCD_4DGL uLCD(D1, D0, D2);
Serial pc(USBTX, USBRX);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
EventQueue queue_uLCD(32 * EVENTS_EVENT_SIZE);

Thread thread_DNN;
Thread thread_uLCD;

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
int mode_index;
int song_index;
/*****************************************************************************/
/*********DNN FUNCTIONS*********/
void initial();
void DNN();
int PredictGesture(float* output);

int main(int argc, char* argv[]) {
    initial();
    thread_DNN.start(&DNN);
    while (true) {
        wait(1);
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
            uLCD.printf("%d", gesture_index);
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