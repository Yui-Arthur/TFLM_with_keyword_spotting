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


"""dataset define"""
class google_speech_commands_dataset(Dataset):
    wav_size : int = None
    num_class : int = None
    label_to_id : dict = None

    def __init__(self , file_list : list[Path]):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        wav_data , sample_rate = torchaudio.load(self.file_list[idx])
        
        # print(self.file_list[idx])
        # print(torchaudio.info(self.file_list[idx]))
        if(wav_data.shape[1] != self.wav_size):
            pad = torch.zeros((1,self.wav_size - wav_data.shape[1]))
            wav_data = torch.cat((wav_data, pad), dim = 1)
        # print(wav_data.shape)
        
        label = self.file_list[idx].parent.stem
        id = torch.tensor([self.label_to_id[label]])
        
        return wav_data , torch.nn.functional.one_hot(id , num_classes=self.num_class).squeeze()

"""model define"""
class attention_model(torch.nn.Module):
    def __init__(self , input_dim = 16000 , out_class = 31 , device = "cpu"):
        super().__init__()
        self.device = device
    
        self.pre_cnn = torch.nn.Sequential(
            # input = 16000
            torch.nn.Conv1d(in_channels=1 , out_channels=32 , kernel_size=32 , stride=4 , padding=15),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32 , out_channels=64 , kernel_size=32 , stride=4 , padding=15),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64 , out_channels=128 , kernel_size=32 , stride=4 , padding=15),
            torch.nn.ReLU(),
            # out 128 , 250
            # torch.nn.Conv1d(in_channels=128 , out_channels=128 , kernel_size=3 , stride=1 , padding=1 , groups=128),
            # torch.nn.Conv1d(in_channels=128 , out_channels=256 , kernel_size=1 , stride=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv1d(in_channels=256 , out_channels=256 , kernel_size=3 , stride=1 , padding=1 , groups=256),
            # torch.nn.Conv1d(in_channels=256 , out_channels=512 , kernel_size=1 , stride=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv1d(in_channels=512 , out_channels=512 , kernel_size=3 , stride=1 , padding=1 , groups=512),
            # torch.nn.Conv1d(in_channels=512 , out_channels=1024 , kernel_size=3 , stride=1),
            # torch.nn.ReLU(),
            # torch.nn.AvgPool1d(kernel_size=8 , stride=4 , padding=2),
            # torch.nn.AdaptiveAvgPool2d((1000,1)),
            
        )

        self.conformer = torchaudio.models.Conformer(input_dim=250 , num_heads=2 , ffn_dim=256 , num_layers=1 , depthwise_conv_kernel_size=31,dropout=0.2)
        # 250 , 128
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=128 , stride=128)

        
        
        # self.avgpool = 
        self.pred_layer = torch.nn.Sequential(
            # torch.nn.Linear(1000 , 512),
            # torch.nn.ReLU(),        
            torch.nn.Linear(250, 256),
            torch.nn.ReLU(),        
            torch.nn.Linear(256, out_class),
        )

    def __conformer_block__(self,x):
        
        conformer_length = (torch.ones(x.shape[0]) * x.shape[1]).to(self.device)
        out , _= self.conformer(x,conformer_length)
        out = torch.transpose(out , dim0=1 , dim1=2)
        out = self.avg_pool(out)
        return out

    def forward(self , x):
        out = self.pre_cnn(x)
        out = self.__conformer_block__(out)
        out = self.pred_layer(out.squeeze())
        return out

"""gen label name to id dict"""
def gen_label_to_id_dict(speech_commands_root_folder):
    sorted_label = sorted([label.stem for label in speech_commands_root_folder.glob("*") if label.is_dir() and label.stem != "_background_noise_"])
    return  {label:idx for idx , label in enumerate(sorted_label)}

"""gen the train / valid / test data list"""
def gen_data_list(speech_commands_root_folder):
    all_data = [class_ for class_ in speech_commands_root_folder.rglob("*.wav") if not class_.match("_background_noise_/*.wav") ]

    with open(speech_commands_root_folder / "validation_list.txt") as f:
        valid_file = [speech_commands_root_folder / _.strip('\n') for _ in f.readlines()]
    with open(speech_commands_root_folder / "testing_list.txt") as f:
        test_file = [speech_commands_root_folder  /_.strip('\n')  for _ in f.readlines()]

    train_file = set(all_data) - (set(valid_file) | set(test_file))
    train_file = list(train_file)

    google_speech_commands_dataset.label_to_id = gen_label_to_id_dict(speech_commands_root_folder)

    print("all data" , len(all_data))

    print(f"train {len(train_file)} , {len(train_file) * 100 / len(all_data):.2f}%")
    print(f"valid {len(valid_file)} , {len(valid_file) * 100 / len(all_data):.2f}%")
    print(f"teat  {len(test_file)} , {len(test_file) * 100 / len(all_data):.2f}%")

    print("label name => id")
    for k , v in google_speech_commands_dataset.label_to_id.items():
        print(f"{k:8}: {v:2}" , end=" , ")
        if (v+1) % 5 == 0 : print()

    random.shuffle(train_file)
    random.shuffle(valid_file)
    random.shuffle(test_file)

    return train_file , valid_file , test_file

"""gen train / valid / test dataloader"""
def gen_dataloader(speech_commands_root_folder , batch_size , workers , logger : logging.Logger , pick_num : tuple[int] = (int(1e9), int(1e9), int(1e9))):
    train_data , valid_data , test_data = gen_data_list(speech_commands_root_folder)
    train_dataset = google_speech_commands_dataset(train_data[:pick_num[0]])
    valid_dataset = google_speech_commands_dataset(valid_data[:pick_num[1]])
    test_dataset = google_speech_commands_dataset(test_data[:pick_num[2]])
    logger.info(f"train : {len(train_dataset)} valid : {len(valid_dataset)} test : {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset , batch_size=batch_size , shuffle=True , drop_last=True , num_workers=workers , pin_memory=True) 
    valid_dataloader = DataLoader(valid_dataset , batch_size=batch_size , num_workers=workers , pin_memory=True) 
    test_dataloader = DataLoader(test_dataset , batch_size=batch_size , num_workers=workers , pin_memory=True)
    
    del train_data , valid_data , test_data , train_dataset , valid_dataset , test_dataset
    return train_dataloader , valid_dataloader , test_dataloader

"""train and valid model with one epoch , return train/valid acc/loss"""
def train_one_epoch(model , train_dataloader , valid_dataloader , max_acc , logger : logging.Logger , root_dir : Path):
    
    # train
    model.train()
    epoch_train_loss = []
    epoch_train_acc = np.array([])
    for data , label in tqdm(train_dataloader):
        pred = model(data.to(device))
        label = label.to(device)
        loss = criterion(pred , label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        result = torch.argmax(pred , dim=1).eq(torch.argmax(label , dim=1)).to("cpu").numpy()
        # log the loss & acc
        epoch_train_acc = np.append(epoch_train_acc , result)
        epoch_train_loss.append(loss.to("cpu").item())
    
    logger.info(f"\ttrain :\tacc = {np.mean(epoch_train_acc):.2f}\tloss = {np.mean(epoch_train_loss):.2f}")

    # valid
    model.eval()
    epoch_valid_loss = []
    epoch_valid_acc = np.array([])
    with torch.no_grad():
        for data , label in tqdm(valid_dataloader):
            pred = model(data.to(device))
            label = label.to(device)
            result = torch.argmax(pred , dim=1).eq(torch.argmax(label , dim=1)).to("cpu").numpy()
            loss = criterion(pred , label.float())

            # log the loss & acc
            epoch_valid_acc = np.append(epoch_valid_acc , result)
            epoch_valid_loss.append(loss.to("cpu").item())

    logger.info(f"\tvalid :\tacc = {np.mean(epoch_valid_acc):.2f}\tloss = {np.mean(epoch_valid_loss):.2f}")
    if(np.mean(epoch_valid_acc) > max_acc):
        logger.info(f"\tupdate best model with acc {np.mean(epoch_valid_acc):.2f}")
        torch.save(model, root_dir / "best_model.pt")

    return np.mean(epoch_train_acc) , np.mean(epoch_train_loss) , np.mean(epoch_valid_acc) , np.mean(epoch_valid_loss)

def logger_setting(root):
    root.mkdir(exist_ok=True)
    logger = logging.getLogger(f"{__name__}")
    log_format = logging.Formatter(f'[%(asctime)s] - %(message)s')
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(filename=root / "run.log", encoding='utf-8' , mode="w")
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

def testing_model(device , test_dataloader , logger : logging.Logger , root_dir : Path = None , model_path : Path = None):
    if root_dir is not None :
        model = torch.load(root_dir / "best_model.pt")
    elif model_path is not None:
        model = torch.load(model_path)
    else:
        raise AttributeError("Testing models root_dir and model_path are not given")
    model.device = device
    model.eval()
    testing_acc = np.array([])
    logger.info("Testing Model")
    with torch.no_grad():
        for data , label in tqdm(test_dataloader):
            pred = model(data.to(device))
            label = label.to(device)
            result = torch.argmax(pred , dim=1).eq(torch.argmax(label , dim=1)).to("cpu").numpy()
            testing_acc = np.append(testing_acc , result)
    
    logger.info(f"acc = {np.mean(testing_acc):.2f}")
    del model

def pt_convert_onnx(pt_path , onnx_path):
    model = torch.load(pt_path).to("cpu")
    model.device = "cpu"
    model.eval()

    torch_input = torch.rand(3, 1, 16000 , dtype=torch.float32).to("cpu")
    model(torch_input)
    
    onnx_program = torch.onnx.export(
        model, 
        torch_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0:"batch_size"},
                      "output" : {0 : "batch_size"}})

# init setting
input_dim = 16000
output_class = 35

# dataset class setting
google_speech_commands_dataset.wav_size = input_dim
google_speech_commands_dataset.num_class = output_class

if __name__ == "__main__":
    # hyperparameter
    epochs = 10
    batch_size = 128
    device = "cuda"
    learning_rate = 1e-3
    workers = 0

    # path & logger setting
    speech_commands_root_folder = Path("./speech_commands")
    root_folder = Path("conformer/1219")
    logger : logging.Logger = logger_setting(root_folder)

    # model init
    model = attention_model(input_dim , output_class , device).to(device)
    optimizer = torch.optim.Adam(model.parameters() , lr = learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    summary(model)
    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f"Model Parameters : {num_params}")
    
    # get dataloader
    train_dataloader , valid_dataloader , test_dataloader = gen_dataloader(speech_commands_root_folder , batch_size , workers , logger , (1000,100,100))
    # torch_input = torch.rand(1, 1, 16000 , dtype=torch.float32).to(device)
    # model(torch_input)
    # train_dataloader , valid_dataloader , test_dataloader = gen_dataloader(speech_commands_root_folder , batch_size , workers)

    # # train / valid
    # train_info = []
    # max_acc = -1
    # for epoch in range(epochs):
    #     logger.info(f"epoch {epoch} :")
    #     epoch_info = train_one_epoch(model , train_dataloader , valid_dataloader , max_acc , logger , root_folder)
    #     train_info.append(list(epoch_info))
    #     max_acc = max(epoch_info[2] ,max_acc)
    # # show train the result    
    # show_train_results(train_info , root_folder)

    # test the model
    # testing_model(device , test_dataloader , logger , root_dir=root_folder)
    # testing_model(device , test_dataloader , logger , model_path="conformer/best_model_86.pt")
    
    # pt_path = Path("conformer/best_model_86.pt")
    pt_path = Path("conformer/1219/best_model.pt")
    pt_convert_onnx(pt_path , onnx_path=pt_path.with_suffix(".onnx"))

# worker
elif __name__ == "__mp_main__":
    speech_commands_root_folder = Path("./speech_commands")
    google_speech_commands_dataset.label_to_id = gen_label_to_id_dict(speech_commands_root_folder)


        
