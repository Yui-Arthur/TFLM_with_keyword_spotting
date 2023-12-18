import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import numpy as np

# dataset
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

# model
class attention_model(torch.nn.Module):
    def __init__(self , input_dim = 16000 , out_class = 31):
        super().__init__()
        # self.pre_cnn = torch.nn.Sequential(
        #     # input = batch , 1 , 16000 , out = batch 3 , 4000
        #     torch.nn.Conv1d(in_channels=1 , out_channels=3 , kernel_size=8 , stride=4 , padding=3),
        #     torch.nn.ReLU(),
        #     # input = batch , 3 , 4000 , out = batch , 3 , 1000
        #     torch.nn.MaxPool1d(kernel_size=8 , stride=4 , padding=2),
        #     # # input = batch , 3 , 1000 , out = batch , 21 , 1000
        #     torch.nn.Conv1d(in_channels=3 , out_channels=21 , kernel_size=3 , stride=1 , padding=1),
        #     torch.nn.ReLU(),
        # )

        
        # # input = batch , 1000 , 21 , out = batch , 1000 , 21
        # self.attention = torch.nn.MultiheadAttention(embed_dim=1000 , num_heads=2 , dropout=0.2 , batch_first=True)
        # # input = batch , 1000 , 21 , out = batch , 1000 , 7
        # self.attention_pool = torch.nn.MaxPool1d(kernel_size=7 , stride=3 , padding=2)

        # self.pred_layer = torch.nn.Sequential(
        #     # input = batch , 1000 , 7 , out = batch , out_class , 7
        #     torch.nn.Linear(1000 , out_class),
        #     torch.nn.ReLU(),
        # )

        self.pre_cnn = torch.nn.Sequential(
            # input = 16000
            torch.nn.Conv1d(in_channels=1 , out_channels=32 , kernel_size=32 , stride=4 , padding=15),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32 , out_channels=64 , kernel_size=32 , stride=4 , padding=15),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64 , out_channels=128 , kernel_size=32 , stride=4 , padding=15),
            torch.nn.ReLU(),
            # out 
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

        self.pred_layer = torch.nn.Sequential(
            # torch.nn.Linear(1000 , 512),
            # torch.nn.ReLU(),        
            torch.nn.Linear(250, 256),
            torch.nn.ReLU(),        
            torch.nn.Linear(256, out_class),
        )
    def __attention_block__(self , x):
        out = self.attention(x,x,x)[0]
        # print(out.shape)
        out = torch.transpose(out , 1 , 2)
        out = self.attention_pool(out)
        return out

    def __conformer_block__(self,x):
        
        conformer_length = np.ones(x.shape[0]) * x.shape[1]
        conformer_length = torch.tensor(conformer_length).to("cuda")
        out = self.conformer(x,conformer_length)
        out = torch.nn.AdaptiveAvgPool2d((250,1))(out[0])
        return out

    def __pred_fc_block__(self , x):
        out = torch.max(x , dim=2).values
        out = self.pred_layer(out)
        return out

    def forward(self , x):
        out = self.pre_cnn(x)
        out = self.__conformer_block__(out)
        out = self.pred_layer(out.squeeze())
        # out = torch.transpose(out , 1 , 2)
        # out = self.__attention_block__(out)   
        # out = self.__pred_fc_block__(out)
        return out

# build dataset
def gen_label_to_id_dict(speech_commands_root_folder):
    sorted_label = sorted([label.stem for label in speech_commands_root_folder.glob("*") if label.is_dir() and label.stem != "_background_noise_"])
    return  {label:idx for idx , label in enumerate(sorted_label)}

def gen_data_list(speech_commands_root_folder):
    all_data = [class_ for class_ in speech_commands_root_folder.rglob("*.wav") if not class_.match("_background_noise_/*.wav") ]

    with open(speech_commands_root_folder / "validation_list.txt") as f:
        valid_data = [speech_commands_root_folder / _.strip('\n') for _ in f.readlines()]
    with open(speech_commands_root_folder / "testing_list.txt") as f:
        testing_data = [speech_commands_root_folder  /_.strip('\n')  for _ in f.readlines()]

    training_data = set(all_data) - (set(valid_data) | set(testing_data))
    training_data = list(training_data)

    google_speech_commands_dataset.label_to_id = gen_label_to_id_dict(speech_commands_root_folder)

    print("all data" , len(all_data))

    print(f"train {len(training_data)} , {len(training_data) * 100 / len(all_data):.2f}%")
    print(f"valid {len(valid_data)} , {len(valid_data) * 100 / len(all_data):.2f}%")
    print(f"teat  {len(testing_data)} , {len(testing_data) * 100 / len(all_data):.2f}%")

    print("label name => id")
    for k , v in google_speech_commands_dataset.label_to_id.items():
        print(f"{k:8}: {v:2}" , end=" , ")
        if (v+1) % 5 == 0 : print()

    return training_data , valid_data , testing_data


def train_one_epoch(model , train_dataloader , valid_dataloader , max_acc):
    
    # train
    model.train()
    epoch_train_loss = []
    epoch_train_acc = []
    for data , label in tqdm(train_dataloader):
        pred = model(data.to(device))
        label = label.to(device)
        loss = criterion(pred , label.float())
        # loss = criterion(pred , label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        result = torch.argmax(pred , dim=1).eq(torch.argmax(label , dim=1)).to("cpu").numpy()
        # log the loss & acc
        epoch_train_acc.append( result[result == True].shape[0] / len(result))
        epoch_train_loss.append(loss.to("cpu").item())
    
    print(f"train :\tacc = {np.mean(epoch_train_acc):.2f}\tloss = {np.mean(epoch_train_loss):.2f}")

    # valid
    model.eval()
    epoch_valid_loss = []
    epoch_valid_acc = []
    with torch.no_grad():
        for data , label in tqdm(valid_dataloader):
            pred = model(data.to(device))
            label = label.to(device)
            result = torch.argmax(pred , dim=1).eq(torch.argmax(label , dim=1)).to("cpu").numpy()

            # log the loss & acc
            epoch_valid_acc.append(result[result == True].shape[0] / len(result))
            epoch_valid_loss.append(loss.to("cpu").item())

    print(f"valid :\tacc = {np.mean(epoch_valid_acc):.2f}\tloss = {np.mean(epoch_valid_loss):.2f}")
    if(np.mean(epoch_valid_acc) > max_acc):
        print(f"update best model with acc {np.mean(epoch_valid_acc):.2f}")
        torch.save(model.state_dict(), "./best_model.pt")
    return np.mean(epoch_train_acc) , np.mean(epoch_train_loss) , np.mean(epoch_valid_acc) , np.mean(epoch_valid_loss)


# init setting
input_dim = 16000
output_class = 35

# dataset class setting
google_speech_commands_dataset.wav_size = input_dim
google_speech_commands_dataset.num_class = output_class

if __name__ == "__main__":
    # hyperparameter
    epochs = 30
    batch_size = 128
    device = "cuda"
    learning_rate = 1e-3
    speech_commands_root_folder = Path("./speech_commands")

    # model init
    model = attention_model(input_dim , output_class).to(device)
    summary(model)
    optimizer = torch.optim.Adam(model.parameters() , lr = learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_data , valid_data , test_data = gen_data_list(speech_commands_root_folder)
    train_dataset = google_speech_commands_dataset(train_data)
    valid_dataset = google_speech_commands_dataset(valid_data)
    train_dataloader = DataLoader(train_dataset , batch_size=batch_size , shuffle=True , drop_last=True , num_workers=6 , pin_memory=True) 
    valid_dataloader = DataLoader(valid_dataset , batch_size=batch_size , num_workers=6 , pin_memory=True) 

    # model(train_dataset[0][0].unsqueeze(dim=0).to(device))
    # exit()
    del train_data , valid_data , test_data , train_dataset , valid_dataset

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    max_acc = 0

    for epoch in range(epochs):
        print(f"epoch {epoch} :")
        epoch_train_acc , epoch_train_loss , epoch_valid_acc , epoch_train_loss = train_one_epoch(model , train_dataloader , valid_dataloader , max_acc)
        max_acc = max(epoch_valid_acc,max_acc)
            

# worker
elif __name__ == "__mp_main__":
    speech_commands_root_folder = Path("./speech_commands")
    google_speech_commands_dataset.label_to_id = gen_label_to_id_dict(speech_commands_root_folder)


        
