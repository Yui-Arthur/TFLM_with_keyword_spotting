
import onnx2tf

def onnx_convert_tf(onnx_path , tf_path):
    onnx2tf.convert(
        overwrite_input_shape=[1,1,16000],
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        
        # param_replacement_file="conformer/param_replacement.json"
    )

onnx_convert_tf("conformer/1219/best_model.onnx" , "conformer/1219/best_model.tf")