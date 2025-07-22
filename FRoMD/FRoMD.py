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

satd_types = {0: 'NON-SATD', 1: 'DESIGN DEBT', 2: 'IMPLEMENTATION DEBT', 3: 'DEFECT DEBT'}

if hasattr(sys, '_MEIPASS'):
    tokenizer_path = os.path.join(sys._MEIPASS, 'roberta/tokenizer.json')
    model_path = os.path.join(sys._MEIPASS, 'models/PMFRoM.onnx')
else:
    tokenizer_path = './roberta/tokenizer.json'
    model_path = './models/PMFRoM.onnx'

def preprocess(comment: str) -> str:
    comment_pattern = compile(r'//|/\*|\*/|\*')
    nonchar_pattern = compile(r'[^\w\s.,!?;:\'\"\-\[\]\(\)@]')
    space_pattern = compile(r'\s{2,}')
    hyphen_pattern = compile(r'-{2,}')
    
    comment = comment_pattern.sub(' ', comment)
    comment = space_pattern.sub(' ', comment)
    comment = nonchar_pattern.sub(' ', comment)
    comment = hyphen_pattern.sub(' ', comment)
    
    return comment.strip().lower()

def load_tokenizer():
    return Tokenizer.from_file(tokenizer_path)

def load_onnx_model():
    device = get_device()
    print(f"\n[System] Loading model to {device}")
    providers = ['CUDAExecutionProvider'] if get_device() == 'GPU' else ['CPUExecutionProvider']
    return InferenceSession(model_path, providers=providers)

def tokenize_batch(texts, max_length=128):
    encodings = tokenizer.encode_batch(texts)
    input_ids = []
    attention_mask = []

    for enc in encodings:
        ids = enc.ids[:max_length]
        mask = enc.attention_mask[:max_length]

        pad_len = max_length - len(ids)
        input_ids.append(ids + [0] * pad_len)
        attention_mask.append(mask + [0] * pad_len)

    return {
        'input_ids': array(input_ids, dtype=int64),
        'attention_mask': array(attention_mask, dtype=int64)
    }

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

def predict_comments(tokenizer, session, texts, batch_size=32):
    input_name = session.get_inputs()[0].name
    mask_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    predictions, probabilities = [], []

    for i in simple_progress_bar(range(0, len(texts), batch_size), desc="[System] Detecting SATD"):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenize_batch(batch_texts)
        outputs = session.run([output_name], {
            input_name: encodings['input_ids'].astype(int64),
            mask_name: encodings['attention_mask'].astype(int64)
        })[0]

        probs = softmax(outputs)
        preds = argmax(probs, axis=1)
        pred_probs = max(probs, axis=1)
        predictions.extend(preds.tolist())
        probabilities.extend(pred_probs.tolist())

    return predictions, probabilities

def softmax(logits):
    e_x = exp(logits - max(logits, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def run_detection(records, tokenizer, session):
    texts = [preprocess(r['comment']) for r in records]
    preds, probs = predict_comments(tokenizer, session, texts)
    for idx, r in enumerate(records):
        r['prediction'], r['probability'] = satd_types[preds[idx]], probs[idx]
    return records

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

    start = time()
    records = [{'filepath': None, 'comment': t} for t in texts]
    results = run_detection(records, tokenizer, session)

    print("\n==== Classification Results ====")
    for idx, r in enumerate(results):
        print(f"{idx+1}. {r['comment']}")
        print(f"   → Prediction: {r['prediction']} (prob={r['probability']:.4f})\n")

def scan_mode(tokenizer, session):
    while True:
        folder = input("[System] Enter directory path to scan: ").strip()
        if folder and os.path.exists(folder):
            break
        print("[Error] Valid directory required.")

    start = time()
    records = scan_files(folder)
    if not records:
        print("[System] No comments found.")
        return

    results = run_detection(records, tokenizer, session)
    out_csv = os.path.join(folder, 'detection_result.csv')
    with open(out_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = DictWriter(f, fieldnames=['filepath', 'comment', 'prediction', 'probability'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[System] Results saved to {out_csv}")

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

def jsonl_mode(tokenizer, session):
    print("[Info] The file must have one JSON object per line, and each object must contain the key 'comment'.")

    while True:
        jsonl_path = input("[System] Enter path to JSONL file: ").strip()
        if jsonl_path and os.path.exists(jsonl_path):
            break
        print("[Error] Valid file path required.")

    start = time()
    records = load_jsonl(jsonl_path)
    if not records:
        print("[Error] No valid records found in JSONL. Please ensure each line has a 'comment' key.")
        return

    print(f"[System] Loaded {len(records)} comments from {jsonl_path}")

    results = run_detection(records, tokenizer, session)

    out_path = jsonl_path.rsplit('.', 1)[0] + '_detection_result.jsonl'
    save_jsonl(results, out_path)
    print(f"[System] Results saved to {out_path}")

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
