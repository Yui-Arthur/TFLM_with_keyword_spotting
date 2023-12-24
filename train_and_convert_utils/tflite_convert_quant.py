import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import logging
from .tf_train import google_speech_commands_dataset , logger_setting


def save_model_tflite_quant(save_model_folder : Path , dataset , logger:logging.Logger):

    def representative_dataset():
        idx = 0
        for data , label in dataset:
            if idx > 1000:
                break
            idx += 1
            
            yield [data]

    # float
    converter = tf.lite.TFLiteConverter.from_saved_model(str(save_model_folder))
    tflite_model = converter.convert()
    open(save_model_folder / 'float_model.tflite', 'wb').write(tflite_model)
    logger.info("Successfully convert tflite float")

    # quant
    converter = tf.lite.TFLiteConverter.from_saved_model(str(save_model_folder))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()
    open(save_model_folder / 'quant_model.tflite', 'wb').write(tflite_quant_model)
    logger.info("Successfully convert tflite quant")
    
def test_tflite_model(tflite_model_path, test_dataset , logger:logging.Logger):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
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
    # init setting
    logger = logger_setting(Path("conformer/1219") , "convert_and_quant")
    input_dim = (16000,1)
    wav_size = 16000
    output_class = 35
    speech_commands_root_folder = Path("./speech_commands")

    _, _, test_dataset = google_speech_commands_dataset(speech_commands_root_folder , wav_size, 1, logger, (1,1,5000), True)
    save_model_tflite_quant(Path("conformer/1219/model") , test_dataset, logger)
    # test_tflite_model("conformer/1219/model/quant.tflite", test_dataset , logger)
    # test_tflite_model("conformer/1219/model/float.tflite", test_dataset , logger)