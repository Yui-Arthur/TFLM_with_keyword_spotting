import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import logging
import sys   
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from conformer_tf import ConformerBlock

"""dataset define"""
def google_speech_commands_dataset(speech_commands_path, wav_size , batch_size , pick_num : tuple[int] = (int(1e9), int(1e9), int(1e9))):
    train_list , valid_list , test_list , label_to_id = gen_data_list(speech_commands_path)

    def __getitem__(file_name):
        file_name = file_name.numpy().decode('ascii')
        wav_data , sample_rate = tf.audio.decode_wav(tf.io.read_file(file_name))
        # print("ori" , wav_data.shape)
        if(wav_data.shape[0] != wav_size):
            pad = np.zeros((wav_size - wav_data.shape[0], 1))
            wav_data = np.concatenate((wav_data , pad) ,axis=0)
        # print("aft" , wav_data.shape)
        

        
        label = Path(file_name).parent.stem
        id = tf.one_hot(label_to_id[label], len(label_to_id))   
        return wav_data , id

    train_dataset = tf.data.Dataset.from_tensor_slices(train_list[:pick_num[0]])
    train_dataset = train_dataset.shuffle(len(train_list[:pick_num[0]]), reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(lambda x: tf.py_function(__getitem__, [x] , [tf.float32 , tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_list[:pick_num[1]])
    valid_dataset = valid_dataset.map(lambda x: tf.py_function(__getitem__, [x] , [tf.float32 , tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test_list[:pick_num[1]])
    test_dataset = test_dataset.map(lambda x: tf.py_function(__getitem__, [x] , [tf.float32 , tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset , valid_dataset , test_dataset

"""model define"""
class attention_model(tf.keras.Model):
    def __init__(self , input_dim = (16000,1) , out_class = 35 , learning_rate=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.out_class = out_class
        self.learning_rate = learning_rate

        self._model = self._BuildModel()
        self._learner = self._BuildLearner()
    
    @tf.function
    def call(self, x:tf.Tensor, training:bool=False) -> tf.Tensor:
        output = self._model(x, training=training)
        return output
    
    @tf.function
    def Train(self, x:tf.Tensor, y:tf.Tensor):
        with tf.GradientTape() as tape:
            output = self.__call__(x, training=True)
            classLoss = self._learner["get_loss"](output, y)

        cGradients = tape.gradient(classLoss, self._model.trainable_variables)
        self._learner["optimize"].apply_gradients(zip(cGradients, self._model.trainable_variables))

    @tf.function
    def Validate(self, x:tf.Tensor, y:tf.Tensor) -> tf.Tensor:
        output = self.__call__(x, training=False)
        review = tf.math.in_top_k(tf.math.argmax(y,axis=1), output, 1)
        classLoss = self._learner["get_loss"](output, y)
        perf = tf.math.reduce_mean(tf.cast(review, dtype="float32"))
        return perf , classLoss

    def _BuildModel(self) -> tf.keras.Model:

        input_tensor = tf.keras.Input(shape=(16000,1))
        feature_map = input_tensor
        feature_map = tf.keras.layers.Conv1D(32, [32], strides=[4], padding="causal")(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)
        feature_map = tf.keras.layers.Conv1D(64, [32], strides=[4], padding="causal")(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)
        feature_map = tf.keras.layers.Conv1D(128, [32], strides=[4], padding="causal")(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)
        feature_map = tf.keras.layers.Conv1D(256, [32], strides=[1], padding="same")(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)
        feature_map = tf.transpose(feature_map , perm = [0,2,1])
        feature_map = tf.keras.layers.AveragePooling1D(256 , strides=256)(feature_map)
        feature_map = tf.squeeze(feature_map , axis=1)
        feature_map = tf.keras.layers.Dense(256, input_dim = 250, activation='relu')(feature_map)
        output_tensor = tf.keras.layers.Dense(self.out_class, input_dim = 256, activation='relu')(feature_map)
        model = tf.keras.Model(input_tensor, output_tensor)
        return model
    
    def _BuildLearner(self) -> dict:
        classLoss = lambda p, y: tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(p+1e-13), axis=1))
        classOptimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        learner = {"get_loss": classLoss, "optimize": classOptimizer}

        return learner
    

"""gen label name to id dict"""
def gen_label_to_id_dict(speech_commands_root_folder):
    sorted_label = sorted([label.stem for label in speech_commands_root_folder.glob("*") if label.is_dir() and label.stem != "_background_noise_"])
    return  {label:idx for idx , label in enumerate(sorted_label)}

"""gen the train / valid / test data list"""
def gen_data_list(speech_commands_root_folder):
    all_data = [str(class_) for class_ in speech_commands_root_folder.rglob("*.wav") if not class_.match("_background_noise_/*.wav") ]

    with open(speech_commands_root_folder / "validation_list.txt") as f:
        valid_file = [str(speech_commands_root_folder / _.strip('\n')) for _ in f.readlines()]
    with open(speech_commands_root_folder / "testing_list.txt") as f:
        test_file = [str(speech_commands_root_folder  /_.strip('\n'))  for _ in f.readlines()]

    train_file = set(all_data) - (set(valid_file) | set(test_file))
    train_file = list(train_file)

    label_to_id = gen_label_to_id_dict(speech_commands_root_folder)

    print("all data" , len(all_data))

    print(f"train {len(train_file)} , {len(train_file) * 100 / len(all_data):.2f}%")
    print(f"valid {len(valid_file)} , {len(valid_file) * 100 / len(all_data):.2f}%")
    print(f"teat  {len(test_file)} , {len(test_file) * 100 / len(all_data):.2f}%")

    print("label name => id")
    for k , v in label_to_id.items():
        print(f"{k:8}: {v:2}" , end=" , ")
        if (v+1) % 5 == 0 : print()

    random.shuffle(train_file)
    random.shuffle(valid_file)
    random.shuffle(test_file)

    return train_file , valid_file , test_file , label_to_id


"""train and valid model with one epoch , return train/valid acc/loss"""
def train_one_epoch(model , train_dataloader , valid_dataloader , max_acc , logger : logging.Logger , root_dir : Path):

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_valid_loss = []
    epoch_valid_acc = []

    for inData, outData in train_dataloader:
        model.Train(inData, outData)
        
    for inData, outData in train_dataloader:
        acc , loss = model.Validate(inData, outData)
        epoch_train_acc.append(acc)
        epoch_train_loss.append(loss)
        
    for inData, outData in valid_dataloader:
        acc , loss = model.Validate(inData, outData)
        epoch_valid_acc.append(acc)
        epoch_valid_loss.append(loss)
        
    epoch_train_acc_mean = tf.math.reduce_mean(epoch_train_acc) * 100
    epoch_valid_acc_mean = tf.math.reduce_mean(epoch_valid_acc) * 100
    epoch_train_loss_mean = tf.math.reduce_mean(epoch_train_loss)
    epoch_valid_loss_mean = tf.math.reduce_mean(epoch_valid_loss)

    logger.info(f"Epoch: {epoch},    Train perf: {epoch_train_acc_mean:.2f},    Valid perf: {epoch_valid_acc_mean:.2f}")

    if(epoch_valid_acc_mean > max_acc):
        logger.info("save best model")
        model.save(root_dir /"model", include_optimizer=False)
    
    return epoch_train_acc_mean, epoch_train_loss_mean , epoch_valid_acc_mean , epoch_valid_loss_mean


def logger_setting(root , log_name):
    root.mkdir(exist_ok=True)
    logger = logging.getLogger(f"{__name__}")
    log_format = logging.Formatter(f'[%(asctime)s] - %(message)s')
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(filename=root / f"{log_name}.log", encoding='utf-8' , mode="w")
    handler.setLevel(logging.DEBUG)   
    handler.setFormatter(log_format)
    logger.addHandler(handler)   

    handler = logging.StreamHandler(sys.stdout)    
    handler.setLevel(logging.DEBUG)                                        
    handler.setFormatter(log_format)    
    logger.addHandler(handler)   
    return logger

def show_train_results(train_info , root_folder : Path):
    train_info = np.array(train_info)

    plt.plot(np.arange(1,train_info.shape[0]+1) , train_info[:,0] , 'r' , label='train')
    plt.plot(np.arange(1,train_info.shape[0]+1) , train_info[:,2] , 'b' , label='valid')
    plt.title("Acc")
    plt.savefig(root_folder / "Acc.png")
    plt.show()

    plt.plot(np.arange(1,train_info.shape[0]+1) , train_info[:,1] , 'r' , label='train')
    plt.plot(np.arange(1,train_info.shape[0]+1) , train_info[:,3] , 'b' , label='valid')
    plt.title("Loss")
    plt.savefig(root_folder / "Loss.png")
    plt.show()

def testing_model(test_dataloader , logger : logging.Logger , root_dir : Path = None , model_path : Path = None):
    if root_dir is not None :
        model = tf.keras.models.load_model(root_dir / "model", compile=False)
    elif model_path is not None:
        model = tf.keras.models.load_model(model_path)
    else:
        raise AttributeError("Testing models root_dir and model_path are not given")
    
    testing_acc = []
    logger.info("Testing Model")
    
    for data , label in tqdm(test_dataloader):
        acc , loss = model.Validate(data , label)
        
        testing_acc.append(acc)
    
    logger.info(f"acc = {np.mean(testing_acc):.2f}")
    del model




if __name__ == "__main__":
    # init setting
    input_dim = (16000,1)
    wav_size = 16000
    output_class = 35
    # hyperparameter
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3

    # path & logger setting
    speech_commands_root_folder = Path("./speech_commands")
    root_folder = Path("conformer/1219")
    logger : logging.Logger = logger_setting(root_folder , "train")

    # model init
    model = attention_model(input_dim , output_class , learning_rate)
    model.build((batch_size,16000,1))
    model._model.summary()
    # exit()
    # logger.info(f"Model Parameters : {num_params}")
    
    # get dataloader
    train_dataloader , valid_dataloader , test_dataloader = google_speech_commands_dataset(speech_commands_root_folder, wav_size, batch_size)
    # train_dataloader , valid_dataloader , test_dataloader = google_speech_commands_dataset(speech_commands_root_folder, wav_size, batch_size , (64,64,64))

    # train / valid
    train_info = []
    max_acc = -1
    for epoch in range(epochs):
        logger.info(f"epoch {epoch} :")
        epoch_info = train_one_epoch(model , train_dataloader , valid_dataloader , max_acc , logger , root_folder)
        train_info.append(list(epoch_info))
        max_acc = max(epoch_info[2] ,max_acc)

    # show train the result    
    show_train_results(train_info , root_folder)

    # test the model
    testing_model(test_dataloader , logger , root_dir=root_folder)
    # testing_model(device , test_dataloader , logger , model_path="conformer/best_model_86.pt")
    
    # pt_path = Path("conformer/best_model_86.pt")
    # pt_path = Path("conformer/1219/best_model.pt")
    # pt_convert_onnx(pt_path , onnx_path=pt_path.with_suffix(".onnx"))


        
