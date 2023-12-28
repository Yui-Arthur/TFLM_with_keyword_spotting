from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import sys   
import matplotlib.pyplot as plt
import random
import tensorflow as tf

"""dataset define"""
def google_speech_commands_dataset(speech_commands_path, wav_size , batch_size, logger:logging.Logger
                                   , pick_num : tuple[int] = (int(1e9), int(1e9), int(1e9)), load_in_memory=False):
    
    train_list , valid_list , test_list , label_to_id = gen_data_list(speech_commands_path)

    logger.info(f"train: {pick_num[0]} , valid: {pick_num[1]} , test: {pick_num[2]} , load_in_memory: {load_in_memory}")

    def __getitem__(file_name, compress=False):
        if type(file_name) != str:
            file_name = file_name.numpy().decode('ascii')

        label = Path(file_name).parent.stem
        if not compress:
            wav_data , sample_rate = tf.audio.decode_wav(tf.io.read_file(file_name))

            if(wav_data.shape[0] != wav_size):
                pad = np.zeros((wav_size - wav_data.shape[0], 1))
                wav_data = np.concatenate((wav_data , pad) ,axis=0)
                    
            id = tf.one_hot(label_to_id[label], len(label_to_id))   

            return wav_data , id
        else:
            int16_wav_data = tf.io.read_file(file_name)
            id = label_to_id[label]
            
            return int16_wav_data , id
    
    def __decompress__(int16_wav_data, id):
        wav_data , sample_rate = tf.audio.decode_wav(int16_wav_data)
        if(wav_data.shape[0] != wav_size):
            pad = np.zeros((wav_size - wav_data.shape[0], 1))
            wav_data = np.concatenate((wav_data , pad) ,axis=0)
            
        id = tf.one_hot(id, len(label_to_id))   
        return wav_data, id

    data_lists = [train_list, valid_list, test_list]
    dataset_lists = [[],[],[]]
    idx_str = ["train" , "valid" , "test"]
    if load_in_memory:
        for idx in range(len(data_lists)):
            logger.info(f"load {idx_str[idx]} dataset with {pick_num[idx]}")
            loaded_data_x = []
            loaded_data_y = []
            for f in tqdm(data_lists[idx][:pick_num[idx]]):
                x , y = __getitem__(f, compress=True) 
                loaded_data_x.append(x)
                loaded_data_y.append(y)
            
            dataset = tf.data.Dataset.from_tensor_slices((loaded_data_x, loaded_data_y))
            dataset = dataset.shuffle(len(loaded_data_x), reshuffle_each_iteration=True)
            dataset = dataset.map(lambda x,y: tf.py_function(__decompress__, [x, y] , [tf.float32 , tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            dataset_lists[idx] = dataset
    else:
        for idx in range(len(data_lists)):
            dataset = tf.data.Dataset.from_tensor_slices(data_lists[idx][:pick_num[idx]])
            dataset = dataset.shuffle(len(data_lists[idx][:pick_num[idx]]), reshuffle_each_iteration=True)
            dataset = dataset.map(lambda x: tf.py_function(__getitem__, [x] , [tf.float32 , tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            dataset_lists[idx] = dataset

    return dataset_lists[0] , dataset_lists[1] , dataset_lists[2]

"""model define"""
class conv_model(tf.keras.Model):
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
            classLoss = self._learner["get_loss"](y, output)
            review = tf.math.in_top_k(tf.math.argmax(y,axis=1), output, 1)
            perf = tf.math.reduce_mean(tf.cast(review, dtype="float32"))

        cGradients = tape.gradient(classLoss, self._model.trainable_variables)
        self._learner["optimize"].apply_gradients(zip(cGradients, self._model.trainable_variables))
        return perf, classLoss


    @tf.function
    def Validate(self, x:tf.Tensor, y:tf.Tensor) -> tf.Tensor:
        output = self.__call__(x, training=False)
        classLoss = self._learner["get_loss"](y , output)
        review = tf.math.in_top_k(tf.math.argmax(y,axis=1), output, 1)
        perf = tf.math.reduce_mean(tf.cast(review, dtype="float32"))
        return perf , classLoss

    def _BuildModel(self) -> tf.keras.Model:

        input_tensor = tf.keras.Input(shape=(16000,1))
        feature_map = input_tensor
        feature_map = tf.keras.layers.Conv1D(32 , [160], strides=[32], padding="causal")(feature_map)
        feature_map = tf.keras.layers.BatchNormalization()(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)

        feature_map = tf.keras.layers.Conv1D(32, [50], strides=[2], padding="causal")(feature_map)
        feature_map = tf.keras.layers.BatchNormalization()(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)

        feature_map = tf.keras.layers.Conv1D(64, [3], strides=[1], padding="causal")(feature_map)
        feature_map = tf.keras.layers.BatchNormalization()(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)

        feature_map = tf.keras.layers.Conv1D(64, [3], strides=[1], padding="causal")(feature_map)
        feature_map = tf.keras.layers.BatchNormalization()(feature_map)
        feature_map = tf.keras.layers.ReLU()(feature_map)


        feature_map = tf.transpose(feature_map , perm = [0,2,1])
        feature_map = tf.keras.layers.AveragePooling1D(64 , strides=64)(feature_map)
        feature_map = tf.squeeze(feature_map , axis=1)
        output_tensor = tf.keras.layers.Dense(self.out_class, input_dim = 64, activation='relu')(feature_map)
        # output_tensor = tf.keras.layers.Dense(self.out_class, input_dim = 256, activation='relu')(feature_map)
        model = tf.keras.Model(input_tensor, output_tensor)
        return model
    
    def _BuildLearner(self) -> dict:
        # classLoss = lambda y, p: tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(p+1e-13), axis=1))
        classLoss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
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

    for inData, outData in tqdm(train_dataloader):
        acc , loss = model.Train(inData, outData)
        epoch_train_acc.append(acc)
        epoch_train_loss.append(loss)
        
    for inData, outData in tqdm(valid_dataloader):
        acc , loss = model.Validate(inData, outData)
        epoch_valid_acc.append(acc)
        epoch_valid_loss.append(loss)
    
    epoch_train_acc_mean = tf.math.reduce_mean(epoch_train_acc) * 100
    epoch_valid_acc_mean = tf.math.reduce_mean(epoch_valid_acc) * 100
    epoch_train_loss_mean = tf.math.reduce_mean(epoch_train_loss)
    epoch_valid_loss_mean = tf.math.reduce_mean(epoch_valid_loss)

    logger.info(f"  Train Acc: {epoch_train_acc_mean:.2f}, Loss: {epoch_train_loss_mean:.2f}")
    logger.info(f"  Valid Acc: {epoch_valid_acc_mean:.2f}, Loss: {epoch_valid_loss_mean:.2f}")

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
    
    logger.info(f"acc = {np.mean(testing_acc)*100:.2f}")
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
    model = conv_model(input_dim , output_class , learning_rate)
    model.build((batch_size,16000,1))
    model._model.summary()
    # exit()
    # logger.info(f"Model Parameters : {num_params}")
    
    # get dataloader
    # train_dataloader , valid_dataloader , test_dataloader = google_speech_commands_dataset(speech_commands_root_folder, wav_size, batch_size , logger)
    train_dataloader , valid_dataloader , test_dataloader = google_speech_commands_dataset(speech_commands_root_folder, wav_size, batch_size , logger, (2000,2000,256) , load_in_memory=True)

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


        
