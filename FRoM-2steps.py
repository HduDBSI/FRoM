import torch
import torch.nn as nn
from transformers import RobertaConfig, AutoModel
from torch.utils.data import DataLoader
import models 
import os
import argparse
from tqdm import tqdm
from CustomDataset import build_dataset
from CustomMetrics import cal_metrics, PRF4TgtCls
import random
import numpy as np
import time
from CCUS import CCUS

os.environ["TOKENIZERS_PARALLELISM"] = "true"
def get_parser():
    parser = argparse.ArgumentParser(description="Fine-tuning Roberta with Multi-Task Learning for SATD Detection")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--token_max_length', type=int, default=128)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--model_name', type=str, default='FRoM.pth')
    parser.add_argument('--folder', type=str, default='data/VG_data')
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=3)
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

def save_model(model_name, encoder, clf):
    model = {'encoder': encoder.state_dict(), 'clf': clf.state_dict()}
    torch.save(model, 'models/'+model_name)

def load_model(model_name, encoder, clf):
    content = torch.load('models/'+model_name, map_location=lambda storage, loc:storage)
    encoder.load_state_dict(content['encoder'])
    clf.load_state_dict(content['clf'])
    return encoder, clf

def train(model_name, train_set, valid_set, encoder, encoder_optimizer, clf, clf_optimizer, loss_fn):
    pbar = tqdm(range(args.epoch_num), total=args.epoch_num, ncols=100, unit="epoch", colour="red")

    best_f1 = 0.1
    train_losses = []
    for epoch in pbar:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
        encoder.train()
        clf.train()
        total_loss = 0

        for inputs in dataloader:
            encoder_optimizer.zero_grad()
            clf_optimizer.zero_grad()

            token, mask, label = inputs['input_ids'].to(args.device), inputs['attention_mask'].to(args.device), inputs['label'].to(args.device)

            outputs1 = encoder(input_ids=token, attention_mask=mask)
            outputs2 = encoder(input_ids=token, attention_mask=mask)
            embed1, embed2 = outputs1['last_hidden_state'][:,0,:], outputs2['last_hidden_state'][:,0,:]
            
            logit1, logit2 = clf(embed1), clf(embed2)
            loss = loss_fn(logit1, logit2, label)
            
            loss.backward()
            encoder_optimizer.step()
            clf_optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(list(dataloader))
        train_losses.append(train_loss)
        valid_f1 = evaluate(valid_set, encoder, clf, valid=True)
        
        if best_f1 < valid_f1:
            best_f1 = valid_f1
            save_model(model_name, encoder, clf)
        pbar.set_postfix({"train_loss": "{:.4f}".format(train_loss), 'valid_f1': "{:.4f}".format(valid_f1), 'best_valid_f1': "{:.4f}".format(best_f1)})

    return train_losses

def evaluate(input_set, encoder, clf, valid=True):
    dataloader = DataLoader(input_set, batch_size=args.batch_size, shuffle=False)

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
    
    metrics = cal_metrics(y_true, y_pred, y_pred_prob, False)
    
    if not valid:
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return y_pred
    
    return metrics['MacroF'] if 'MacroF' in metrics else metrics['F']

t = time.time()
parser = get_parser()
args = parser.parse_args()

args_dict = {arg: getattr(args, arg) for arg in vars(args)} 
print(*[f"{k}: {v}" for k, v in args_dict.items()], sep='\n') 
print()

set_seed(args.seed)

config = RobertaConfig.from_pretrained('roberta/config.json')
loss_fn = models.CustomLoss(args.weight) if args.MTL else nn.CrossEntropyLoss()

#########################first step######################
encoder = AutoModel.from_pretrained('roberta/', config=config).to(args.device)
bin_clf = models.Classifier(args.embed_dim, args.dropout, 2).to(args.device)
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
bin_clf_optimizer = torch.optim.Adam(bin_clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_set = build_dataset(f'{args.folder}/preprocessed/train.jsonl', args.token_max_length, 2)
valid_set = build_dataset(f'{args.folder}/preprocessed/valid.jsonl', args.token_max_length, 2)
test_set = build_dataset(f'{args.folder}/preprocessed/test.jsonl', args.token_max_length, 2)

ccus = CCUS(device=args.device, model_name='CCUS', class_num=2, threshold=args.threshold, seed=args.seed)
train_set = ccus.fit_resample(train_set, valid_set)

train("bin_model", train_set, valid_set, encoder, encoder_optimizer, bin_clf, bin_clf_optimizer, loss_fn)
encoder, bin_clf = load_model("bin_model", encoder, bin_clf)
bin_y_pred = evaluate(test_set, encoder, bin_clf, valid=False)


#########################second step######################
encoder = AutoModel.from_pretrained('roberta/', config=config).to(args.device)
mul_clf = models.Classifier(args.embed_dim, args.dropout, 3).to(args.device)
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
mul_clf_optimizer = torch.optim.Adam(mul_clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_set = build_dataset(f'{args.folder}/preprocessed/train.jsonl', args.token_max_length, 3)
valid_set = build_dataset(f'{args.folder}/preprocessed/valid.jsonl', args.token_max_length, 3)
test_set = build_dataset(f'{args.folder}/preprocessed/test.jsonl', args.token_max_length, 4)

train("mul_model", train_set, valid_set, encoder, encoder_optimizer, mul_clf, mul_clf_optimizer, loss_fn)
encoder, mul_clf = load_model("mul_model", encoder, mul_clf)
mul_y_pred = evaluate(test_set, encoder, mul_clf, valid=False)

## evaluate
mul_y_pred = mul_y_pred + 1
mul_y_pred[bin_y_pred == 0] = 0

y_pred = mul_y_pred

y_true = test_set.label.numpy()

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
print('cost time:', time.time()-t)
