from onnxruntime import InferenceSession, get_device
from tokenizers import Tokenizer
from numpy import int64, max, argmax, exp, array
from time import time
from re import compile, findall
from csv import DictWriter
import os
import sys
import json
print(f"[System] Initializing")

SATD_TYPES = {0: 'NON-SATD', 1: 'DESIGN DEBT', 2: 'IMPLEMENTATION DEBT', 3: 'DEFECT DEBT'}

TOKENIZER_PATH = os.path.join(sys._MEIPASS, 'tokenizer.json') if hasattr(sys, '_MEIPASS') else 'tokenizer.json'
MODEL_PATH = os.path.join(sys._MEIPASS, 'FineTunedModel.onnx') if hasattr(sys, '_MEIPASS') else 'FineTunedModel.onnx'

# ---------------- Load Tokenizer & Model ----------------
def load_tokenizer(path=TOKENIZER_PATH):
    return Tokenizer.from_file(path)

def load_onnx_model(path=MODEL_PATH):
    device = get_device()
    print(f"[System] Loading model to {device}")
    providers = ['CUDAExecutionProvider'] if device == 'GPU' else ['CPUExecutionProvider']
    return InferenceSession(path, providers=providers)

# ---------------- Text Preprocessing ----------------
def preprocess(comment: str) -> str:
    comment = compile(r'//|/\*|\*/|\*').sub(' ', comment)
    comment = compile(r'[^\w\s.,!?;:\'\"\-\[\]\(\)@]').sub(' ', comment)
    comment = compile(r'\s{2,}').sub(' ', comment)
    comment = compile(r'-{2,}').sub(' ', comment)
    return comment.strip().lower()

# ---------------- Tokenization ----------------
def tokenize_batch(tokenizer, texts, max_length=128):
    encodings = tokenizer.encode_batch(texts)
    input_ids, attention_mask = [], []

    for enc in encodings:
        ids, mask = enc.ids[:max_length], enc.attention_mask[:max_length]
        pad_len = max_length - len(ids)
        input_ids.append(ids + [0] * pad_len)
        attention_mask.append(mask + [0] * pad_len)

    return {
        'input_ids': array(input_ids, dtype=int64),
        'attention_mask': array(attention_mask, dtype=int64)
    }

# ---------------- Progress Bar ----------------
def simple_progress_bar(iterable, total=None, desc="", length=25):
    total = total or len(iterable)
    start_time = time()
    for i, item in enumerate(iterable, 1):
        done = int(length * i / total)
        percent = (i / total) * 100
        elapsed = time() - start_time
        sys.stdout.write(
            f"\r{desc} [{'#' * done}{'.' * (length - done)}] {percent:.1f}% - {elapsed:.1f}s elapsed"
        )
        sys.stdout.flush()
        yield item
    print()

# ---------------- Prediction ----------------
def predict_comments(tokenizer, session, texts, batch_size=32):
    def softmax(logits):
        e_x = exp(logits - max(logits, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    input_name, mask_name = session.get_inputs()[0].name, session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    all_preds, all_probs = [], []

    for i in simple_progress_bar(range(0, len(texts), batch_size), desc="[System] Detecting SATD"):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenize_batch(tokenizer, batch_texts)
        outputs = session.run([output_name], {
            input_name: encodings['input_ids'].astype(int64),
            mask_name: encodings['attention_mask'].astype(int64)
        })[0]

        probs = softmax(outputs)   # shape (batch, 4)
        preds = argmax(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist()) 

    return all_preds, all_probs

# ---------------- Run Detection ----------------
def run_detection(records, tokenizer, session):
    texts = [preprocess(r['comment']) for r in records]
    preds, all_probs = predict_comments(tokenizer, session, texts)
    for idx, r in enumerate(records):
        r['prediction'] = SATD_TYPES[preds[idx]]
        r['prob_NON-SATD'] = all_probs[idx][0]
        r['prob_DESIGN'] = all_probs[idx][1]
        r['prob_IMPLEMENTATION'] = all_probs[idx][2]
        r['prob_DEFECT'] = all_probs[idx][3]
    return records

# ---------------- Interactive Mode ----------------
def interactive_mode(tokenizer, session):
    print("[System] Enter comments to classify (empty line to finish):")
    texts = []
    while True:
        line = input("> ").strip()
        if line == "":
            break
        texts.append(line)

    if not texts:
        print("[Info] No input received, exiting interactive mode.")
        return

    records = [{'filepath': None, 'comment': t} for t in texts]
    results = run_detection(records, tokenizer, session)

    print("\n==== Classification Results ====")
    print("[Info] The numbers after the prediction are probabilities for [NON-SATD, DESIGN, IMPLEMENTATION, DEFECT].")
    for idx, r in enumerate(results):
        print(f"{idx+1}. {r['comment']}")
        print(f"   → Prediction: {r['prediction']} "
          f"[{r['prob_NON-SATD']:.4e}, {r['prob_DESIGN']:.4e}, "
          f"{r['prob_IMPLEMENTATION']:.4e}, {r['prob_DEFECT']:.4e}]\n")

# ---------------- Scan Mode ----------------
def scan_mode(tokenizer, session):
    def extract_comments_from_java(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"[Warning] Cannot read {file_path}: {e}")
            return []
        return [c.strip() for c in findall(r'//.*|/\*[\s\S]*?\*/', content) if c.strip()]

    def scan_files(folder):
        records, file_count = [], 0
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.java'):
                    file_count += 1
                    full_path = os.path.join(root, file)
                    comments = extract_comments_from_java(full_path)
                    for comment in comments:
                        records.append({'filepath': full_path, 'comment': comment})
        print(f"[System] Found {file_count} .java files, extracted {len(records)} comments.")
        return records

    while True:
        folder = input("[System] Enter directory path to scan: ").strip()
        if folder and os.path.exists(folder):
            break
        print("[Error] Valid directory required.")

    records = scan_files(folder)
    if not records:
        print("[System] No comments found.")
        return

    results = run_detection(records, tokenizer, session)
    out_csv = os.path.join(folder, 'detection_result.csv')
    with open(out_csv, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ['filepath', 'comment', 'prediction',
                    'prob_NON-SATD', 'prob_DESIGN', 'prob_IMPLEMENTATION', 'prob_DEFECT']
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[System] Results saved to {out_csv}")

# ---------------- JSONL Mode ----------------
def jsonl_mode(tokenizer, session):
    def load_jsonl(path):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'comment' in obj:
                        records.append(obj)
                    else:
                        print("[Warning] line missing 'comment' field, skipping.")
                except json.JSONDecodeError:
                    print("[Warning] invalid json line, skipping.")
        return records
    
    def save_jsonl(records, out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("[Info] The file must have one JSON object per line, and each object must contain the key 'comment'.")

    while True:
        jsonl_path = input("[System] Enter path to JSONL file: ").strip()
        if jsonl_path and os.path.exists(jsonl_path):
            break
        print("[Error] Valid file path required.")

    records = load_jsonl(jsonl_path)
    if not records:
        print("[Error] No valid records found in JSONL. Please ensure each line has a 'comment' key.")
        return

    print(f"[System] Loaded {len(records)} comments from {jsonl_path}")

    results = run_detection(records, tokenizer, session)

    out_path = jsonl_path.rsplit('.', 1)[0] + '_detection_result.jsonl'
    save_jsonl(results, out_path)
    print(f"[System] Results saved to {out_path}")

# ---------------- Main Menu ----------------
def main_menu(tokenizer, session):
    while True:
        print("======= SATD Comment Classifier Menu =======")
        print("1. Interactive mode (analyze individual comments)")
        print("2. Scan mode (analyze all .java files in directory)")
        print("3. JSONL mode (analyze comments from a jsonl file)")
        print("4. Exit")

        mode = input("\n[System] Select mode (1/2/3/4): ").strip()
        os.system('cls' if os.name == 'nt' else 'clear')
        if mode == '1':
            interactive_mode(tokenizer, session)
        elif mode == '2':
            scan_mode(tokenizer, session)
        elif mode == '3':
            jsonl_mode(tokenizer, session)
        elif mode == '4':
            print("[System] Exiting program. Goodbye!")
            break
        else:
            print("[Error] Please enter 1, 2, 3, or 4.")
        input("\n[System]Press Enter to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    session = load_onnx_model()

    main_menu(tokenizer, session)
