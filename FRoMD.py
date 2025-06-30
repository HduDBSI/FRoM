# To install PyInstaller: 
#   pip install pyinstaller
#
# To build executable: 
#   pyinstaller --onefile --hidden-import torch --hidden-import transformers --hidden-import models FRoMD.py
#
# Make sure the following files are in the same directory as the executable or correctly referenced in your code:
#   - models/PMFRoM.pth
#   - roberta/config.json
#   - roberta/pytorch_model.bin
#   - roberta/tokenizer_config.json
#   - roberta/tokenizer.json
#   - roberta/vocab.json
#
# The required RoBERTa-related files can be downloaded from: 
#   https://huggingface.co/FacebookAI/roberta-base/tree/main
#
# After building, the executable will be generated in the `dist/` directory.
# Example to run the executable in Ubuntu:
#   cd dist
#   chmod +x FRoMD
#   ./FRoMD --device cpu --text "TODO: refactor this" "buggy code"
# Or run interactively:
#   ./FRoMD --device cpu
#   # Then enter comments line by line, empty line to finish input.

import re
import torch
from transformers import AutoTokenizer, AutoModel, RobertaConfig
from torch.utils.data import DataLoader, Dataset
import time
import torch.nn as nn
import argparse

satd_types = {0: 'NON-SATD', 1: 'DESIGN DEBT', 2: 'IMPLEMENTATION DEBT', 3: 'DEFECT DEBT'}

class Classifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1, class_num: int = 1):
        super(Classifier, self).__init__() 

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, class_num),
        )

        self.reset_parameter()

    def reset_parameter(self):  
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=0, mode='fan_in', nonlinearity='relu')
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, emd):
        return self.mlp(emd)
    
class CustomDataset(Dataset):
    def __init__(self, texts, max_length=128):
        tokenizer = AutoTokenizer.from_pretrained('roberta/')
        outputs = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        self.input_ids = outputs['input_ids']
        self.attention_mask = outputs['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
    
def preprocess(comment: str) -> str:
    comment_pattern = re.compile(r'//|/\*|\*/|\*')
    nonchar_pattern = re.compile(r'[^\w\s.,!?;:\'\"\-\[\]\(\)@]')
    space_pattern = re.compile(r'\s{2,}')
    hyphen_pattern = re.compile(r'-{2,}')
    
    comment = comment_pattern.sub(' ', comment)
    comment = space_pattern.sub(' ', comment)
    comment = nonchar_pattern.sub(' ', comment)
    comment = hyphen_pattern.sub(' ', comment)
    
    return comment.strip().lower()

def load_model(device: str='cpu'):
    config = RobertaConfig.from_pretrained('roberta/config.json')
    encoder = AutoModel.from_pretrained('roberta/', config=config).to(device)
    clf = Classifier(input_dim=768, dropout=0.1, class_num=4).to(device)

    content = torch.load('models/PMFRoM.pth', map_location=lambda storage, loc:storage)
    encoder.load_state_dict(content['encoder'])
    clf.load_state_dict(content['clf'])

    return encoder, clf

def detect(encoder, clf, dataloader, device):
    encoder.eval()
    clf.eval()

    y_pred, y_pred_prob = [], []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = encoder(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
            embed = outputs['last_hidden_state'][:,0,:]
            
            prob = torch.softmax(clf(embed), dim=1)
            y_pred += prob.argmax(dim=1).cpu().numpy().tolist()
            y_pred_prob += (1 - prob[:,0]).cpu().numpy().tolist()
    
    
    return y_pred, y_pred_prob

def classify_texts(texts, device):
    texts = [preprocess(text) for text in texts]
    dataset = CustomDataset(texts)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    encoder, clf = load_model(device)
    y_pred, y_pred_prob = detect(encoder, clf, dataloader, device)
    
    results = []
    for i in range(len(texts)):
        results.append((texts[i], satd_types[y_pred[i]], y_pred_prob[i]))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SATD comment classifier")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu, cuda, or cuda:0')
    parser.add_argument('--text', type=str, nargs='*', help='Code comments to classify')
    args = parser.parse_args()

    if not args.text:
        print("Enter comments to classify (one per line, empty line to end):")
        texts = []
        while True:
            line = input()
            if not line.strip():
                break
            texts.append(line)
    else:
        texts = args.text

    t_time = time.time()
    results = classify_texts(texts, args.device)

    for text, satd_type, prob in results:
        print(f"{text} : {satd_type} (prob={prob:.4f})")

    print(f"Time elapsed: {time.time() - t_time:.2f} seconds")
