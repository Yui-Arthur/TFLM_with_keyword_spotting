import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import logging
from train_tf import google_speech_commands_dataset , logger_setting


def save_model_tflite_quant(save_model_folder , dataset , logger):

    def representative_dataset():
        dataset = [(i,i) for i in range(100)]
        for data , label in dataset:
            data = np.random.rand(1, 16000, 1).astype(np.float32)
            yield [data]

    # float
    converter = tf.lite.TFLiteConverter.from_saved_model(save_model_folder)
    tflite_model = converter.convert()
    open(Path(save_model_folder) / 'float_model.tflite', 'wb').write(tflite_model)

    # int8
    converter = tf.lite.TFLiteConverter.from_saved_model(save_model_folder)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()
    open(Path(save_model_folder) / 'quant_model.tflite', 'wb').write(tflite_quant_model)
    
def test_tflite_model(tflite_model_path, test_dataset , logger:logging.Logger):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input = interpreter.get_input_details()[0]
    output = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input['quantization']
    output_scale, output_zero_point = output["quantization"]
    
    logger.info(f"testing with {tflite_model_path}")
    
    testing_acc = np.array([])
    for data , label in tqdm(test_dataset):
        if input["dtype"] == np.float32:
            interpreter.set_tensor(input['index'], data)
        else:
            quant_data = data.numpy() / input_scale + input_zero_point
            interpreter.set_tensor(input['index'], quant_data.astype(input["dtype"] ))

        interpreter.invoke()

        if output["dtype"] == np.float32:
            pred = interpreter.get_tensor(output['index'])
        else:
            dequant_pred = interpreter.get_tensor(output['index'])
            pred = (dequant_pred - output_zero_point) * output_scale

        testing_acc = np.append(testing_acc, np.argmax(pred) == np.argmax(label))   
    
    logger.info(f"result acc = {np.mean(testing_acc)*100:.2f}")

if __name__ == "__main__":
    # save_model_tflite_quant("conformer/1219/model" , None , None)
    logger = logger_setting(Path("conformer/1219") , "convert_and_quant")
    # init setting
    input_dim = (16000,1)
    wav_size = 16000
    output_class = 35
    speech_commands_root_folder = Path("./speech_commands")

    _, _, test_dataset = google_speech_commands_dataset(speech_commands_root_folder , wav_size, 1, logger, (1,1,100), True)
    test_tflite_model("conformer/1219/model/quant.tflite", test_dataset , logger)
    test_tflite_model("conformer/1219/model/float.tflite", test_dataset , logger)