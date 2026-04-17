import torch 
import pandas as pd
from torchmetrics import Accuracy, ConfusionMatrix
from tqdm import tqdm
from statistics import mean

class PenguinDataset(torch.utils.data.Dataset):
    
    def __init__(self, file_path, label_column):
        
        self.data = pd.read_csv(file_path)
        
        # only use the columns that have information value
        self.labels = self.data[[label_column]]
        self.features = self.data.drop(label_column, axis=1)
        self.features = self.features.iloc[:, 1:] # drop the first column (index)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.features.iloc[idx]
        labels = self.labels.iloc[idx]
        features = torch.tensor(features.values, dtype=torch.float)
        # using datatype long for the label is necessary for CrossEntropyLoss
        labels = torch.tensor(labels.values, dtype=torch.long)
        
        return features, labels
         
        
class PenguinModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=0.3):
        super(PenguinModel, self).__init__()

        # We first increase the dimensions to create a high-dimensional representation
        # in_dim = number of features
        self.fc1 = torch.nn.Linear(in_dim, 256)
        self.fc2 = torch.nn.Linear(256, 512)

        # We then shrink this high-dimensional representation to get a prediction
        self.fc3 = torch.nn.Linear(512, 128)

        # out_dim = number of classese
        self.fc4 = torch.nn.Linear(128, out_dim)

        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        # no activation function for the last layer since CrossEntropyLoss expects unnormalized logits.
        x = self.fc4(x)
        
        return x 

def eval_on_dataset(model_for_eval, dataloader):
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=3).to(device) 

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for features, labels in dataloader:  
            features, labels = features.to(device), labels.to(device)
            labels = torch.squeeze(labels)

            # since our model outputs unnormalized values we need to apply softmax for predictions
            # softmax over dim=1 --> output.shape= (batch_size, 3)
            outputs = torch.softmax(model_for_eval(features), dim=1)

            accuracy.update(outputs, labels)
            confusion_matrix.update(outputs, labels)

    acc = accuracy.compute()
    cm = confusion_matrix.compute()

    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
        
        
### Config ### 

data_path = "penguin_dataset.csv"
train_perc = 0.7
val_perc = 0.15
test_perc = 0.15

num_epochs = 1500
batch_size = 32
learing_rate = 1e-4
test = True

##############

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu") 
# Running the code on CPU helps you to debug errors!
# device = torch.device("cpu")
print(f"Using device: {device}")
###############

whole_dataset = PenguinDataset(data_path, label_column="species")

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, 
                                                                         [train_perc, val_perc, test_perc], 
                                                                         torch.Generator().manual_seed(42)
                                                                         )

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = PenguinModel(in_dim=6, out_dim=3)

optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
loss_criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)
for epoch in range(num_epochs):
    epoch_loss = []
    for features, labels in train_dataloader:
        
        optimizer.zero_grad()
        
        features, labels = features.to(device), labels.to(device)
        
        # We squeeze one dimension because CrossEntropyLoss expects the labels (target) to be of size (batch_size)
        # before squeezing: labels.shape = (batch_size, 1)
        labels = torch.squeeze(labels)
        # after squeezing: labels.shape = (batch_size)
        
        outputs = model(features)
        
        # shapes: outputs -> (batch_size, 3); labels -> (batch_size)
        loss = loss_criterion(outputs, labels)
        epoch_loss.append(loss.item())
        
        # Calculate gradients and make one update of model weights
        loss.backward()
        optimizer.step()

    print(f"Avg loss of epoch {epoch}: {mean(epoch_loss)}")
        
       
print("Finished Training. Evaluating on validation...")
eval_on_dataset(model, val_dataloader)

if test:
    print("Evaluating on test dataset")
    eval_on_dataset(model, test_dataloader)
