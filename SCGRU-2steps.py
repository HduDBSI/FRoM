import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from re import sub
import json
import argparse
from tqdm import tqdm
import random
import argparse
from CustomMetrics import cal_metrics, PRF4TgtCls
from torch.utils.data import DataLoader
import time

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='data/VG_data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--epoch_num',type=int, default=20)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--seq_len',type=int, default=128)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='scgru.pth')
    parser.add_argument('--seed', type=int, default=0)
    
    return parser

def set_seed(seed):
    random.seed(seed)  # Randomness for Python
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed for reproducibility
    np.random.seed(seed)  # Randomness for numpy
    torch.manual_seed(seed)  # Randomness for torch CPU, set seed for CPU
    torch.cuda.manual_seed(seed)  # Randomness for torch GPU, set seed for current GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU, set seed for all GPUs
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark mode for deterministic behavior
    torch.backends.cudnn.deterministic = True  # Choose deterministic algorithms

class CNNGRU(nn.Module):
    def __init__(self, embed_dim, class_num, filter_size_list=[3, 4, 5], filter_num=128, drop_out=0.5):
        """
        TextCNN model for text classification.

        Args:
            class_num (int): Number of output classes.
            filter_size_list (list): List of filter sizes for convolutional layers.
            filter_num (int): Number of filters per convolutional layer.
        """
        super(CNNGRU, self).__init__()

        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (filter_size, embed_dim)) for filter_size in filter_size_list])
        self.gru = nn.GRU(filter_num * len(filter_size_list), embed_dim)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        """
        Forward pass for the TextCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, class_num).
        """

        # Add a channel dimension for convolutional layers
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)

        # # Apply convolutional layers and max pooling
        conv_outs = [F.relu(conv(x).squeeze(3)) for conv in self.convs]

        pooled_outs = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outs]

        outs = self.dropout(torch.cat(pooled_outs, 1))
        outs = outs.unsqueeze(0)

        outs, _ = self.gru(outs)

        logits = self.fc(outs.squeeze(0))
        return logits

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = sub(r"[^A-Za-z0-9(),\+!?\'\`]", " ", string)
    string = sub(r"\'s", " \'s", string)
    string = sub(r"\'ve", " \'ve", string)
    string = sub(r"n\'t", " n\'t", string)
    string = sub(r"\'re", " \'re", string)
    string = sub(r"\'d", " \'d", string)
    string = sub(r"\'ll", " \'ll", string)
    string = sub(r",", " , ", string)
    string = sub(r"!", " ! ", string)
    string = sub(r"\(", " \( ", string)
    string = sub(r"\)", " \) ", string)
    string = sub(r"\?", " \? ", string)
    string = sub(r"\+", " \+ ", string)
    string = sub(r"\s{2,}", " ", string)
    return string.strip().lower().split(' ')

def load_data(json_file, class_num):
    comments = []
    labels = []
    with open(json_file, "r") as file:
        for line in file:
            data = json.loads(line)
            if class_num == 2:
                comments.append(clean_str(data["comment"]))
                labels.append(int(data["label"]!=0))
            elif class_num == 3 and data["label"] != 0:
                comments.append(clean_str(data["comment"]))
                labels.append(data["label"]-1)
            elif class_num == 4:
                comments.append(clean_str(data["comment"]))
                labels.append(data["label"])
    return comments, labels

class CustomDataset(Dataset):
    def __init__(self, embeds, labels):
        self.embeds = embeds
        self.labels = labels

    def __getitem__(self, idx):
        return self.embeds[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def pad(sentences, seq_len):
    return [
        sentence[:seq_len] + [pad_idx] * max(0, seq_len - len(sentence))
        for sentence in sentences
    ]

def embed(sentences, embeddings):
    return [
        torch.stack([torch.tensor(embeddings[word], dtype=torch.float) if word in embeddings else unk_embedding for word in sentence])
        for sentence in sentences
    ]

def train(model_name, train_set, valid_set, textcnn, optimizer, loss_fn):
    pbar = tqdm(range(args.epoch_num), total=args.epoch_num, ncols=100, unit="epoch", colour="red")

    best_f1 = 0.1
    train_losses = []
    for epoch in pbar:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
        textcnn.train()

        total_loss = 0

        for embeds, labels in dataloader:
            optimizer.zero_grad()

            embeds = embeds.to(args.device)
            labels = labels.to(args.device)

            logits = textcnn(embeds)
            loss = loss_fn(logits, labels)
        
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(list(dataloader))
        train_losses.append(train_loss)
        valid_f1 = evaluate(valid_set, textcnn, valid=True)

        if best_f1 < valid_f1:
            best_f1 = valid_f1
            save_model(model_name, textcnn)
        pbar.set_postfix({"train_loss": "{:.4f}".format(train_loss), 'valid_f1': "{:.4f}".format(valid_f1), 'best_valid_f1': "{:.4f}".format(best_f1)})
    
    return train_losses

def save_model(model_name, textcnn):
    model = {'textcnn': textcnn.state_dict()}
    torch.save(model, 'models/'+model_name)

def load_model(model_name, textcnn):
    content = torch.load('models/'+model_name, map_location=lambda storage, loc:storage)
    textcnn.load_state_dict(content['textcnn'])
    return textcnn

# test if valid is false
def evaluate(input_set, textcnn, valid=True):

    dataloader = DataLoader(input_set, batch_size=args.batch_size, shuffle=False)

    y_pred, y_true, y_pred_logit = [], [], []
    
    textcnn.eval()

    for embeds, labels in dataloader:
        
        embeds = embeds.to(args.device)

        with torch.no_grad():
            
            logits = textcnn(embeds)
            probs = torch.softmax(logits, dim=1)
            tmp_pred = probs.argmax(dim=1).int().cpu()
            y_pred_positive_prob = (1 - probs[:,0]).cpu()

            y_pred += tmp_pred
            y_true += labels.tolist()
            y_pred_logit += y_pred_positive_prob

    metrics = cal_metrics(y_true, y_pred, y_pred_logit, False)

    if not valid:
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return y_pred
    
    return metrics['MacroF'] if 'MacroF' in metrics else metrics['F']

def load_word_embeddings():
    word_embeddings = {}
    with open("data/cache/glove.6B.300d.txt", 'r') as f:
        for line in f.readlines():
            data = line.split()
            word_embeddings[str(data[0])] = list(map(float, data[1:]))
    return word_embeddings

t = time.time()

parser = get_parser()
args = parser.parse_args()

args_dict = {arg: getattr(args, arg) for arg in vars(args)} 
print(*[f"{k}: {v}" for k, v in args_dict.items()], sep='\n') 
print()

set_seed(args.seed)

pad_idx = '<PAD>'      
unk_embedding = torch.tensor(np.random.uniform(-0.01, 0.01, args.embed_dim), dtype=torch.float)
word_embeddings = load_word_embeddings()
word_embeddings[pad_idx] = [0.0] * 300

#########################first step######################
X_train, y_train = load_data(args.folder+"/preprocessed/train.jsonl", 4)
X_valid, y_valid = load_data(args.folder+"/preprocessed/valid.jsonl", 4)
X_test, y_test = load_data(args.folder+"/preprocessed/test.jsonl", 4)

for class_i in range(1, 4):
    X_train_i, y_train_i = load_data(f'{args.folder}/preprocessed/train_{class_i}.jsonl', 4)
    X_train.extend(X_train_i)
    y_train.extend(y_train_i)

X_train = embed(pad(X_train, args.seq_len), word_embeddings)
X_valid = embed(pad(X_valid, args.seq_len), word_embeddings)
X_test = embed(pad(X_test, args.seq_len), word_embeddings)

bin_y_train = [1 if y != 0 else 0 for y in y_train]
bin_y_valid = [1 if y != 0 else 0 for y in y_valid]
bin_y_test = [1 if y != 0 else 0 for y in y_test]

train_set = CustomDataset(X_train, torch.tensor(bin_y_train))
valid_set = CustomDataset(X_valid, torch.tensor(bin_y_valid))
test_set = CustomDataset(X_test, torch.tensor(bin_y_test))

loss_fn = nn.CrossEntropyLoss() # softmax is inside

textcnn = CNNGRU(args.embed_dim, 2).to(args.device)
optimizer = torch.optim.AdamW(textcnn.parameters(), lr=args.lr, weight_decay=5e-4)

train("bin_SCGRU", train_set, valid_set, textcnn, optimizer, loss_fn)

textcnn = load_model("bin_SCGRU", textcnn)
bin_y_pred = evaluate(test_set, textcnn, valid=False)


#########################second step######################
X_train, y_train = load_data(args.folder+"/preprocessed/train.jsonl", 3)
X_valid, y_valid = load_data(args.folder+"/preprocessed/valid.jsonl", 3)
X_test, y_test = load_data(args.folder+"/preprocessed/test.jsonl", 4)

for class_i in range(1, 4):
    X_train_i, y_train_i = load_data(f'{args.folder}/preprocessed/train_{class_i}.jsonl', 3)
    X_train.extend(X_train_i)
    y_train.extend(y_train_i)

X_train = embed(pad(X_train, args.seq_len), word_embeddings)
X_valid = embed(pad(X_valid, args.seq_len), word_embeddings)
X_test = embed(pad(X_test, args.seq_len), word_embeddings)

train_set = CustomDataset(X_train, torch.tensor(y_train))
valid_set = CustomDataset(X_valid, torch.tensor(y_valid))
test_set = CustomDataset(X_test, torch.tensor(y_test))

textcnn = CNNGRU(args.embed_dim, 3).to(args.device)
optimizer = torch.optim.AdamW(textcnn.parameters(), lr=args.lr, weight_decay=5e-4)

train("mul_SCGRU", train_set, valid_set, textcnn, optimizer, loss_fn)

textcnn = load_model("mul_SCGRU", textcnn)
mul_y_pred = evaluate(test_set, textcnn, valid=False)


## evaluate
mul_y_pred = mul_y_pred + 1
mul_y_pred[bin_y_pred == 0] = 0

y_pred = mul_y_pred

y_true = test_set.labels.numpy()

metrics = {}
# calculate precision, recall and f1-score for class 1
metrics['P4C1'], metrics['R4C1'], metrics['F4C1'] = PRF4TgtCls(y_true, y_pred, 1)

# calculate precision, recall and f1-score for class 2
metrics['P4C2'], metrics['R4C2'], metrics['F4C2'] = PRF4TgtCls(y_true, y_pred, 2)

# calculate precision, recall and f1-score for class 3
metrics['P4C3'], metrics['R4C3'], metrics['F4C3'] = PRF4TgtCls(y_true, y_pred, 3)

metrics['MacroP'] = (metrics['P4C1'] + metrics['P4C2'] + metrics['P4C3']) / 3
metrics['MacroR'] = (metrics['R4C1'] + metrics['R4C2'] + metrics['R4C3']) / 3
metrics['MacroF'] = (metrics['F4C1'] + metrics['F4C2'] + metrics['F4C3']) / 3

for key, value in metrics.items():
    print("{}: {:.4f}".format(key, value))
print('total time:', time.time()-t)
