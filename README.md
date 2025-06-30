# Source Code for the Paper

**"Leveraging Multi-Task Learning to Fine-Tune RoBERTa for Self-Admitted Technical Debt Identification and Classification"**

## Environment

* Python 3.10
* CUDA 11.5

## Installation

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

* `RQ1.py` to `RQ5.py`: Scripts to reproduce the experimental results reported in the paper.
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
Create a `models/` directory in your project root.

Download the model weights `PMFRoM.pth` from
  [https://github.com/HduDBSI/FRoM/releases/tag/V1.0.0](https://github.com/HduDBSI/FRoM/releases/tag/V1.0.0)
  and place the file in:

  ```
  /models
  ```
To classify comments using FRoMD:

```bash
python FRoMD.py --device cpu --text "TODO: refactor this" "buggy code"
```

or run interactively:

```bash
python FRoMD.py --device cpu
```

Then enter comments one per line; finish input with an empty line.

To build a standalone executable (optional):

```bash
pip install pyinstaller
pyinstaller --onefile --hidden-import torch --hidden-import transformers FRoMD.py
```

Run the generated file (in `dist/`):

```bash
cd dist
chmod +x FRoMD
./FRoMD --device cpu --text "TODO: refactor this" "buggy code"
```
