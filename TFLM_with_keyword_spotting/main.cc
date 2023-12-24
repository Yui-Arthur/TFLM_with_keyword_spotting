#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "wav_data.h"
#include "model_data_quant.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mbed.h"

constexpr int kTensorArenaSize = 265000;
uint8_t tensor_arena[kTensorArenaSize];

TfLiteStatus RegisterOps(tflite::MicroMutableOpResolver<7> &op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddExpandDims());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  
  return kTfLiteOk;
}

TfLiteStatus LoadModelAndPerformInference(){
    MicroPrintf("LoadModelAndPerformInference() Start");
    const unsigned char *model_data_ptr = model_data_quant;
    const tflite::Model* model = tflite::GetModel(model_data_ptr);


    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    tflite::MicroMutableOpResolver<7> op_resolver;
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

    

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

    TfLiteTensor* input = interpreter.input(0);
    TFLITE_CHECK_NE(input, nullptr);

    TfLiteTensor* output = interpreter.output(0);
    TFLITE_CHECK_NE(output, nullptr);

    float output_scale = 0, input_scale = 0;
    int output_zero_point = 0 , input_zero_point = 0;

    output_scale = output->params.scale , output_zero_point = output->params.zero_point;
    input_scale = input->params.scale , input_zero_point = input->params.zero_point;
    MicroPrintf("Input quant info : %f , %d",input->params.scale , input->params.zero_point);
    MicroPrintf("Output quant info : %f , %d",output->params.scale , output->params.zero_point);

    // MicroPrintf("%d , %f", wav_data_len , wav_data_array[0]);
    int quant_model_input [wav_data_len];
    for(int i=0; i<wav_data_len; i++){
        quant_model_input[i] = int(wav_data_array[i] / input_scale + input_zero_point);
    }
    std::copy_n(quant_model_input, wav_data_len, interpreter.input(0)->data.int8);

    mbed::Timer t;
    MicroPrintf("Start Inference");
    t.start();
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    t.stop();
    MicroPrintf("End Inference with %ld (ns)" , std::chrono::duration_cast<std::chrono::microseconds>(t.elapsed_time()).count());

    int kCategoryCount=35;
    std::vector<float> category_predictions(kCategoryCount);

    for(int i=0; i<kCategoryCount; i++){

        category_predictions[i] = (interpreter.output(0)->data.int8[i] - output_zero_point) * output_scale;
        MicroPrintf("  %.4f", static_cast<double>(category_predictions[i]));
    }
    int prediction_index =
        std::distance(std::begin(category_predictions),
                    std::max_element(std::begin(category_predictions),
                                     std::end(category_predictions)));
    
    MicroPrintf("pred %i", prediction_index);   
    return kTfLiteOk;


}

int main(int argc, char* argv[]) {
    // MicroPrintf("%d , %f", wav_data_len , wav_data_array[0]);
    tflite::InitializeTarget();

    LoadModelAndPerformInference();
    MicroPrintf("~~~ALL TESTS PASSED~~~\n");
    return kTfLiteOk;
}