from PIL import Image
import argparse
import numpy as np
import subprocess
from pathlib import Path
import wave

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav', type=str, help='wav path' , default="speech_commands/cat/0a2b400e_nohash_0.wav")
    parser.add_argument('--model', type=str, help='model.tflite path')
    parser.add_argument('--project', type=str, help='project path' , default="TFLM_with_keyword_spotting")
    parser.add_argument('--quant', action="store_true")

    opt = parser.parse_args()
    return opt

def convert_wav(opt):
    project_path = Path(opt['project'])
    wav_size = 16000
    wav_data = []
    with wave.open(opt['wav'] , "r") as w:
        frames_size = w.getnframes()
        frames = w.readframes(frames_size)
        max_16bit =(2**15)
        for idx in range(frames_size):
            data_16bit = int.from_bytes(frames[2*idx: 2*idx+2] , byteorder ='little' , signed=True)
            float_num = data_16bit / max_16bit
            wav_data.append(float_num)
    
    with open(project_path / "wav_data.cc" , "w") as f:
        f.write("#include <stdint.h>\n")
        f.write("float wav_data[] = {\n")

        for idx in range(wav_size):
            if len(wav_data) > idx:
                f.write(f"{wav_data[idx]}")
            else:
                f.write(f"0.0")

            if idx != wav_size-1:
                f.write(",\t")
            if (idx+1) % 40 == 0:
                f.write("\n")
        
        f.write("\n};\n")
        f.write("unsigned int wav_data_len = 16000;\n")
                


    with open(project_path / "wav_data.h" , "w") as f:
        f.write("#ifndef WAV_DATA_H_\n")
        f.write("#define WAV_DATA_H_\n")
        f.write("   extern float wav_data[];\n")
        f.write("   extern unsigned int wav_data_len;\n")
        f.write("#endif  // WAV_DATA_H_\n")
        pass

def convert_model(opt):
    project_path = Path(opt['project'])
    if opt['quant']:
        subprocess.run(["/bin/bash", "script/convert_model.sh", '-i', opt['model'] , '-p', str(project_path), '-q'] , timeout=5 )
    else:
        subprocess.run(["/bin/bash", "script/convert_model.sh", '-i', opt['model'] , '-p', str(project_path)] , timeout=5 )
    


def main(opt):
    convert_wav(opt)
    convert_model(opt)



if __name__ == '__main__':
    opt = parse_opt()
    main(vars(opt))