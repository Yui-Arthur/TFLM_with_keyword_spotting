
import onnx2tf
import onnxruntime as ort
import numpy as np
from train import gen_dataloader , logger_setting
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf

def onnx_convert_tf(onnx_path , tflite_path):
    onnx2tf.convert(
        overwrite_input_shape=[1,1,16000],
        input_onnx_file_path=onnx_path,
        output_folder_path=tflite_path,
        copy_onnx_input_output_names_to_tflite=True,
        # non_verbose=True,
    )

def tflite_quant(tflite_path , speech_commands_folder , logger):

    def representative_dataset():
        _, _, test_dataset = gen_dataloader(speech_commands_folder , 32 , 0 , logger , (0,0,1000))
        for data , label in test_dataset:
            data = data.numpy()
            yield data

    converter = tf.lite.TFLiteConverter.from_keras_model()
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()
    open(f'{tflite_path}_quant.tflite', 'wb').write(tflite_quant_model)
    



def onnx_accuracy_check(onnx_path , speech_commands_folder , logger):
    _, _, test_dataset = gen_dataloader(speech_commands_folder , 32 , 0 , logger )
    ort_sess = ort.InferenceSession(onnx_path , providers=["CUDAExecutionProvider"])
    # ort_sess = ort.InferenceSession(onnx_path , providers=["GPUExecutionProvider"])
    print(ort_sess.get_providers())
    
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    output_name = ort_sess.get_outputs()[0].name
    input_shape[0] = 32
    
    testing_acc = np.array([])
    for data , label in test_dataset:
        out =  ort_sess.run([output_name] , {input_name : data.numpy()})[0]
        result = np.argmax(out, axis=1) == np.argmax(label.numpy() , axis=1)
        print(result)
        testing_acc = np.append(testing_acc , result)
        exit()    
    print(np.mean(testing_acc))


    # in_data = np.random.random(input_shape).astype(np.float32)

# print(ort.get_device())
# exit()
if __name__ == "__main__":
    logger = logger_setting(Path("conformer/1221") , "convert_and_quant")
    tflite_quant("conformer/1221/saved_model/best_model_float32.tflite" , Path("speech_commands") , logger)
    # onnx_convert_tf("conformer/1221/best_model.onnx" , "conformer/1221/best_model.tf")
    # onnx_accuracy_check("conformer/1221/best_model.onnx" , Path("speech_commands") , logger)