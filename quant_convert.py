
import onnx2tf
import onnxruntime as ort
import numpy as np
from train import gen_dataloader , logger_setting
from pathlib import Path
from tqdm import tqdm

def onnx_convert_tf(onnx_path , tf_path):
    onnx2tf.convert(
        overwrite_input_shape=[1,1,16000],
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        
        # param_replacement_file="conformer/param_replacement.json"
    )


def onnx_accuracy_check(onnx_path , speech_commands_folder , logger):
    _, _, test_dataset = gen_dataloader(speech_commands_folder , 32 , 0 , logger )
    ort_sess = ort.InferenceSession(onnx_path , providers=["CUDAExecutionProvider"])
    # ort_sess = ort.InferenceSession(onnx_path , providers=["GPUExecutionProvider"])
    print(ort_sess.get_providers())
    exit()
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    output_name = ort_sess.get_outputs()[0].name
    input_shape[0] = 32
    
    testing_acc = np.array([])
    for data , label in test_dataset:
        out =  ort_sess.run([output_name] , {input_name : data.numpy()})[0]
        result = np.argmax(out, axis=1) == np.argmax(label.numpy() , axis=1)
        
        testing_acc = np.append(testing_acc , result)
        
    print(np.mean(testing_acc))


    # in_data = np.random.random(input_shape).astype(np.float32)
# onnx_convert_tf("conformer/1219/best_model.onnx" , "conformer/1219/best_model.tf")

# print(ort.get_device())
# exit()
logger = logger_setting(Path("conformer/1219"))
onnx_accuracy_check("conformer/1219/best_model.onnx" , Path("speech_commands") , logger)