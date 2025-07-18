# To install PyInstaller: 
#   pip install pyinstaller
#
# To build executable: 
#   pyinstaller --onefile FRoMD.py
#
# Make sure the following files are in the same directory as the executable or correctly referenced in your code:
#   - ./models/PMFRoM.pth
#   - ./roberta/config.json
#   - ./roberta/tokenizer_config.json
#   - ./roberta/tokenizer.json
#   - ./roberta/vocab.json
#
# The fine-tuned model can be downloaded from: 
#   https://github.com/HduDBSI/FRoM/releases/download/V1.0.0/PMFRoM.pth

# The required RoBERTa-related files can be downloaded from: 
#   https://huggingface.co/FacebookAI/roberta-base/tree/main or https://github.com/HduDBSI/FRoM/tree/main/roberta
#
# After building, the executable will be generated in the `dist/` directory.
# Example to run the executable in Ubuntu:
#   cd dist
#   chmod +x FRoMD
#   ./FRoMD 

import re
import torch
from transformers import AutoTokenizer, AutoModel, RobertaConfig
from torch.utils.data import DataLoader, Dataset
import time
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm

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
        tokenizer = AutoTokenizer.from_pretrained('./roberta/')
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
    print(f"\n[System] Loading model onto device: {device}...")

    config = RobertaConfig.from_pretrained('./roberta/config.json')
    encoder = AutoModel.from_config(config).to(device)
    clf = Classifier(input_dim=768, dropout=0.1, class_num=4).to(device)

    content = torch.load('./models/PMFRoM.pth', map_location=lambda storage, loc:storage)
    encoder.load_state_dict(content['encoder'])
    clf.load_state_dict(content['clf'])
    print("")
    return encoder, clf

def detect(encoder, clf, dataloader, device):
    encoder.eval()
    clf.eval()

    y_pred, y_pred_prob = [], []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Detecting SATD", unit="batch"):
            outputs = encoder(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
            embed = outputs['last_hidden_state'][:,0,:]
            
            prob = torch.softmax(clf(embed), dim=1)
            y_pred += prob.argmax(dim=1).cpu().numpy().tolist()
            y_pred_prob += (1 - prob[:,0]).cpu().numpy().tolist()
    
    return y_pred, y_pred_prob

def run_detection(records, encoder, clf, device):
    texts = [preprocess(r['comment']) for r in records]
    dataloader = DataLoader(CustomDataset(texts), batch_size=128)
    y_pred, y_prob = detect(encoder, clf, dataloader, device)
    for idx, r in enumerate(records):
        r['prediction'], r['probability'] = satd_types[y_pred[idx]], y_prob[idx]
    return records

def extract_comments_from_java(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[Warning] Cannot read {file_path}: {e}")
        return []
    comments = re.findall(r'//.*|/\*[\s\S]*?\*/', content)
    return [c.strip() for c in comments if c.strip()]

def scan_files(folder):
    all_records, file_count = [], 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.java'):
                file_count += 1
                full_path = os.path.join(root, file)
                comments = extract_comments_from_java(full_path)
                for comment in comments:
                    all_records.append({'filepath': full_path, 'comment': comment})
    print(f"Found {file_count} .java files, extracted {len(all_records)} comments.")
    return all_records

def interactive_mode(encoder, clf, device):
    print("Enter comments to classify (empty line to finish):")
    texts = []
    while True:
        line = input("> ").strip()
        if line == "":
            break
        texts.append(line)

    if not texts:
        print("[Info] No input received, exiting interactive mode.")
        return

    start = time.time()
    records = [{'filepath': None, 'comment': t} for t in texts]
    print(f"\nClassifying {len(records)} comments...\n")

    results = run_detection(records, encoder, clf, device)

    print("==== Classification Results ====")
    for idx, r in enumerate(results):
        print(f"{idx+1}. {r['comment']}")
        print(f"   → Prediction: {r['prediction']} (prob={r['probability']:.4f})\n")

    print(f"Total time: {time.time() - start:.2f}s")

def scan_mode(encoder, clf, device):
    while True:
        folder = input("\nEnter directory path to scan: ").strip()
        if folder and os.path.exists(folder):
            break
        print("[Error] Valid directory required.")
    start = time.time()
    records = scan_files(folder)
    if not records:
        print("No comments found.")
        return
    results = run_detection(records, encoder, clf, device)
    df = pd.DataFrame(results)
    out_csv = os.path.join(folder, 'detection_result.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    print(f"Total time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    print("=== SATD Comment Classifier ===")

    cwd = os.getcwd()
    print(f"Current workspace: {cwd}\n")

    print("Please ensure the following setup before running:")
    print("- Two folders exist in the current directory: models/ and roberta/")
    print("- Download the fine-tuned model PMFRoM.pth from:")
    print("  https://github.com/HduDBSI/FRoM/releases/download/V1.0.0/PMFRoM.pth")
    print("  and place it inside models/")
    print("- Download the following files from:")
    print("  https://github.com/HduDBSI/FRoM/tree/main/roberta")
    print("  and place them inside roberta/:")
    print("    - config.json")
    print("    - tokenizer_config.json")
    print("    - tokenizer.json")
    print("    - vocab.json")

    print("\nExpected directory structure:")
    print(r"""
    .
    ├── FRoMD
    ├── models
    │   └── PMFRoM.pth
    └── roberta
        ├── config.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json
    """)


    print("Please enter processing device:")
    print("1. CPU (default)")
    print("2. GPU (if available)")

    while True:
        device_choice = input("\nSelect device (1/2): ").strip()
        if device_choice in ['1', '2']:
            break
        print("[Error] Please enter 1 or 2")

    device = 'cuda' if device_choice == '2' else 'cpu'

    encoder, clf = load_model(device)

    print("Please select operation mode:")
    print("1. Interactive mode (analyze individual comments)")
    print("2. Scan mode (analyze all .java files in directory)")
    
    while True:
        mode_choice = input("\nSelect mode (1/2): ").strip()
        if mode_choice in ['1', '2']:
            break
        print("[Error] Please enter 1 or 2")
    mode = 'interactive' if mode_choice == '1' else 'scan'

    if mode == 'interactive':
        interactive_mode(encoder, clf, device)

    elif mode == 'scan':
        scan_mode(encoder, clf, device)
