# Source Code for the Paper

**"Leveraging Multi-Task Learning to Fine-Tune RoBERTa for Self-Admitted Technical Debt Identification and Classification"**

## Environment and Installation

* Python 3.10
* CUDA 11.5
  
All dependencies are listed in `requirements.txt`.
You can install them with a single command:

```bash
pip install -r requirements.txt
```

## Code Structure

* `FRoM.py`: The method proposed in this paper.
* `CCUS.py`: The data downsampling module proposed in this paper.
* `SCGRU.py`, `CNN-based.py`, `XGB-based.py`: Source code for the baselines used in our experiments.

## Preparation

1. Before running the code, please **download the RoBERTa model weights** from
   [pytorch\_model.bin](https://huggingface.co/FacebookAI/roberta-base/blob/main/pytorch_model.bin)
   and place the file in the directory:

   ```
   /roberta
   ```

2. If you plan to run `SCGRU.py`, please **download GloVe embeddings** from
   [glove.6B.300d.txt](https://nlp.stanford.edu/data/glove.6B.zip)
   and place the file in:

   ```
   /data/cache/
   ```

## Running Experiments

* `RQ1.py` to `RQ6.py`: Scripts to reproduce the experimental results reported in the paper.
  **Note:** The `/logs` directory contains previous log files.
  If you wish to fully reproduce the experiments on your machine, please clear the `/logs` directory before running these scripts.

## Case Study

The `case.xlsx` file contains data for our case study:

* The scan results from FRoMD on **all 4,799 comments** in an open-source project named Presto
  ([Presto v0.278](https://github.com/prestodb/presto/tree/0.278)).
* Classification results and reasoning for 40 of these comments as provided by **ChatGPT-4o**.
* Manual labels for these 40 comments.
* Links to the corresponding code for these 40 comments.

## FRoMD (FRoM Detector)
### 📥 Download

* **Linux (amd64):**
  [FRoMD-1.0.0-amd64\_linux.zip](https://github.com/HduDBSI/FRoM/releases/download/V1.0.0/FRoMD-1.0.0-amd64_linux.zip)

* **Windows (win64):**
  [FRoMD-1.0.0-win64.zip](https://github.com/HduDBSI/FRoM/releases/download/V1.0.0/FRoMD-1.0.0-win64.zip)

### 🚀 Quick Start

#### Linux

```bash
# Unzip the release
unzip FRoMD-1.0.0-amd64_linux.zip
cd FRoMD-1.0.0

# Grant execution permission
chmod +x FRoMD

# Run the tool
./FRoMD
```

#### Windows

```bash
# After unzipping, simply double-click FRoMD.exe
# Or run from terminal:
FRoMD.exe
```

### 💡 Features

FRoMD offers three operating modes:

* **\[1] Interactive Mode**
  Manually input comments and get instant SATD classification.

* **\[2] Scan Mode**
  Automatically scan all `.java` files in a directory and output a CSV file with classification results.

* **\[3] JSONL Mode**
  Classify comments from a JSONL file (one JSON object per line with a `comment` key) and generate a corresponding JSONL output file.

---

### 📂 Output Files

* **Scan Mode Output:**
  A `detection_result.csv` file will be saved in the target directory.

* **JSONL Mode Output:**
  A `_detection_result.jsonl` file will be generated alongside the input JSONL file.

