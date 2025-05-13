import torch
import torch.nn as nn
from transformers import RobertaConfig, AutoModel
from torch.utils.data import DataLoader
import models 
import os
import argparse
from tqdm import tqdm
from CustomDataset import build_dataset, build_balanced_dataset
from CustomMetrics import cal_metrics
import random
import numpy as np
import time
from CCUS import CCUS

os.environ["TOKENIZERS_PARALLELISM"] = "true"
def get_parser():
    parser = argparse.ArgumentParser(description="Fine-tuning Roberta with Multi-Task Learning for SATD Detection")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--token_max_length', type=int, default=128)
    parser.add_argument('--epoch_num', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--model_name', type=str, default='FRoM.pth')
    parser.add_argument('--folder', type=str, default='data/Maldonado_data')
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--weight', type=float, default=0.4)
    parser.add_argument('--class_balance', type=str, default='CCUS', choices=['CCUS', 'RUS', 'None'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--MTL', type=int, default=1)
    return parser

def set_seed(seed):
    random.seed(seed)  # Randomness for Python
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed for reproducibility
    np.random.seed(seed)  # Randomness for numpy
    torch.manual_seed(seed)  # Randomness for torch CPU, set seed for CPU
    torch.cuda.manual_seed(seed)  # Randomness for torch GPU, set seed for current GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU, set seed for all GPUs
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark mode for deterministic behavior
    # torch.backends.cudnn.deterministic = True  # Choose deterministic algorithms

def save_model():
    model = {'encoder': encoder.state_dict(), 'clf': clf.state_dict()}
    torch.save(model, 'models/'+args.model_name)

def load_model():
    content = torch.load('models/'+args.model_name, map_location=lambda storage, loc:storage)
    encoder.load_state_dict(content['encoder'])
    clf.load_state_dict(content['clf'])

def train():
    pbar = tqdm(range(args.epoch_num), total=args.epoch_num, ncols=100, unit="epoch", colour="red")

    best_f1 = 0.3
    train_losses = []
    for epoch in pbar:
        train_loss = train_one_epoch()
        train_losses.append(train_loss)
        valid_f1 = evaluate(valid=True)
        if best_f1 < valid_f1:
            best_f1 = valid_f1
            save_model()
        pbar.set_postfix({"train_loss": "{:.4f}".format(train_loss), 'valid_f1': "{:.4f}".format(valid_f1), 'best_valid_f1': "{:.4f}".format(best_f1)})

    return train_losses

def train_one_epoch():
    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    encoder.train()
    clf.train()
    total_loss = 0

    for inputs in dataloader:
        encoder_optimizer.zero_grad()
        clf_optimizer.zero_grad()

        token, mask, label = inputs['input_ids'].to(args.device), inputs['attention_mask'].to(args.device), inputs['label'].to(args.device)

        if args.MTL:
            outputs1 = encoder(input_ids=token, attention_mask=mask)
            outputs2 = encoder(input_ids=token, attention_mask=mask)
            embed1, embed2 = outputs1['last_hidden_state'][:,0,:], outputs2['last_hidden_state'][:,0,:]
            
            logit1, logit2 = clf(embed1), clf(embed2)
            loss = loss_fn(logit1, logit2, label)
        else:
            outputs = encoder(input_ids=token, attention_mask=mask)
            embed = outputs['last_hidden_state'][:,0,:]

            logit = clf(embed)
            loss = loss_fn(logit, label)

        loss.backward()
        encoder_optimizer.step()
        clf_optimizer.step()
        total_loss += loss.item()

    total_loss = total_loss / len(list(dataloader))
    return total_loss

# test if valid is false
def evaluate(valid=True):
    dataloader = DataLoader(valid_set if valid else test_set, batch_size=args.batch_size, shuffle=False)

    y_pred, y_true, y_pred_prob = [], [], []
    encoder.eval()
    clf.eval()

    for inputs in dataloader:
        with torch.no_grad():
            input_ids, mask = inputs['input_ids'].to(args.device), inputs['attention_mask'].to(args.device)
            outputs = encoder(input_ids=input_ids, attention_mask=mask)
            embed = outputs['last_hidden_state'][:,0,:]

            prob = torch.softmax(clf(embed), dim=1) 

            y_pred += prob.argmax(dim=1).cpu()
            y_true += inputs['label'].tolist()
            y_pred_prob += (1 - prob[:,0]).cpu()
    
    metrics = cal_metrics(y_true, y_pred, y_pred_prob, not valid)

    return metrics['MacroF'] if 'MacroF' in metrics else metrics['F']

t = time.time()
parser = get_parser()
args = parser.parse_args()

args_dict = {arg: getattr(args, arg) for arg in vars(args)} 
print(*[f"{k}: {v}" for k, v in args_dict.items()], sep='\n') 
print()

set_seed(args.seed)

train_set = build_dataset(f'{args.folder}/preprocessed/train.jsonl' if args.train_file == None else args.train_file, args.token_max_length, args.class_num)
valid_set = build_dataset(f'{args.folder}/preprocessed/valid.jsonl' if args.valid_file == None else args.valid_file, args.token_max_length, args.class_num)
test_set = build_dataset(f'{args.folder}/preprocessed/test.jsonl' if args.test_file == None else args.test_file, args.token_max_length, args.class_num)

if args.class_balance == 'CCUS':
    ccus = CCUS(device=args.device, model_name='CCUS', class_num=args.class_num, threshold=args.threshold, seed=args.seed)
    train_set = ccus.fit_resample(train_set, valid_set)
elif args.class_balance == 'RUS':
    train_set = build_balanced_dataset(train_set)

config = RobertaConfig.from_pretrained('roberta/config.json')
encoder = AutoModel.from_pretrained('roberta/', config=config).to(args.device)

clf = models.Classifier(args.embed_dim, args.dropout, args.class_num).to(args.device)
loss_fn = models.CustomLoss(args.weight) if args.MTL else nn.CrossEntropyLoss()

encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_losses = train()
load_model()
evaluate(valid=False)
print('cost time:', time.time()-t)