import pyaudio
import numpy as np
import time
import tensorflow as tf
from train_and_convert_utils import gen_label_to_id_dict , save_model_tflite_quant , test_tflite_model , logger_setting , google_speech_commands_dataset
from pathlib import Path

def microphone_callback(in_data, frame_count, time_info, flag):
    global output_stream , two_seconds_data , cur_idx , two_seconds_len

    if cur_idx + len(in_data) > two_seconds_len:
        overflow = cur_idx + len(in_data) - two_seconds_len
        two_seconds_data[cur_idx:] = in_data[:overflow]
        two_seconds_data[:overflow] = in_data[overflow:]
    else:
        two_seconds_data[cur_idx : cur_idx + len(in_data)] = in_data[:]
    
    cur_idx = (cur_idx + len(in_data)) % two_seconds_len

    return (in_data, pyaudio.paContinue)

def tflite_infer(tf_interpreter, data):
    
    data = np.expand_dims(data, 1)
    data = np.expand_dims(data, 0)

    input = tf_interpreter.get_input_details()[0]
    output = tf_interpreter.get_output_details()[0]

    input_scale, input_zero_point = input['quantization']
    output_scale, output_zero_point = output["quantization"]

    if input["dtype"] == np.float32:
        interpreter.set_tensor(input['index'], data)
    else:
        quant_data = data / input_scale + input_zero_point
        interpreter.set_tensor(input['index'], quant_data.astype(input["dtype"] ))

    interpreter.invoke()

    if output["dtype"] == np.float32:
        pred = interpreter.get_tensor(output['index'])
    else:
        dequant_pred = interpreter.get_tensor(output['index'])
        pred = (dequant_pred - output_zero_point) * output_scale

    return pred

def get_pyaudio_input():
    p = pyaudio.PyAudio()

    input_stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input_device_index=2,
                    input=True,
                    stream_callback=microphone_callback)

    return input_stream 
    
    

# for i in range(p.get_device_count()):
#     d = (p.get_device_info_by_index(i))
#     print(d['index'], d['name'], d['maxInputChannels'])
# # exit()

# root_dir = Path("model/12_31_train/model")
# logger = logger_setting(root_dir , "convert_and_quant")
# input_dim = (16000,1)
# wav_size = 16000
# output_class = 35
# speech_commands_root_folder = Path("./speech_commands")

# _, _, test_dataset = google_speech_commands_dataset(speech_commands_root_folder , wav_size, 1, logger, (1,1,1), True)
# # save_model_tflite_quant(root_dir, test_dataset, logger)
# test_tflite_model(root_dir / "quant_model.tflite", test_dataset , logger)
# test_tflite_model(root_dir / "float_model.tflite", test_dataset , logger)
# exit()

if __name__ == '__main__':

    global two_seconds_data, cur_idx

    tflite_model_path = "model/12_31_train/model/float_model.tflite"
    speech_commands_root_folder = Path("./speech_commands")
    two_seconds_len = 16000 * 2 * 2
    int16_max = 2**15
    two_seconds_data = bytearray(two_seconds_len) #[0 for i in range(32000)]
    cur_idx = 0
    id_to_label_dic = {v:k for k,v in gen_label_to_id_dict(speech_commands_root_folder).items()}
    print(id_to_label_dic)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    input_stream = get_pyaudio_input()
    input_stream.start_stream()
    while 1:
        
        time.sleep(0.3)

        one_seconds_len = two_seconds_len // 2
        if cur_idx > one_seconds_len:
            data = np.frombuffer(two_seconds_data[cur_idx-one_seconds_len:cur_idx], dtype=np.int16).astype(np.float32) / int16_max
            # print(tflite_infer(interpreter, data))
            # output_stream.write(bytes(two_seconds_data[cur_idx-one_seconds_len:cur_idx]))
        else:
            prev_idx = one_seconds_len - cur_idx
            data = two_seconds_data[-prev_idx:] + two_seconds_data[:cur_idx]
            data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / int16_max

        result = tflite_infer(interpreter, data)
        # print(result)
        print(np.max(result))
        if(np.max(result) > 0.3):
            print(id_to_label_dic[np.argmax(result)])
        else:
            print("None")
            # output_stream.write(bytes(two_seconds_data[-prev_idx:]))
            # output_stream.write(bytes(two_seconds_data[:cur_idx]))

    input_stream.stop_stream()
    
    