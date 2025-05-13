# multi classification
import os
import time
import subprocess
from RQutils import *

datasets = {
    'dataset-M': 'Maldonado_data/',
    'dataset-VG': 'VG_data/',
}

methods = {
    'CNN-based': 'CNN_based.py',
    'XGB-based': 'XGB_based.py',
    'SCGRU': 'SCGRU.py',
    'FRoM': 'FRoM.py'
}

log_folder = 'logs/RQ2'

rounds = 10
class_num = 4
conda_env = 'pyten'
device = 'cuda:1'

def getTVT(path):
    train_file = 'data/' + path + 'preprocessed/train.jsonl'
    valid_file = 'data/' + path + 'preprocessed/valid.jsonl'
    test_file = 'data/' + path + 'preprocessed/test.jsonl'
    folder = 'data/' + path

    return f'--folder {folder} --train_file {train_file} --valid_file {valid_file} --test_file {test_file}'


for dataset, path in datasets.items():
    for method, pyfile in methods.items():
        for round in range(rounds):
            os.makedirs(log_folder, exist_ok=True)
            log_file = f"{log_folder}/multi_{dataset}_{method}_{round}.txt"
            if os.path.exists(log_file):
                print(f"{log_file} already exists, skipping...")
                continue
            t_round = time.time()
            tvt = getTVT(path)

            if pyfile == 'XGB_based.py':
                command = f"conda run -n {conda_env} python {pyfile} {tvt} --class_num {class_num} --seed {round}"
            else:
                command = f"conda run -n {conda_env} python {pyfile} {tvt} --class_num {class_num} --seed {round} --device {device}"

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, sterr = process.communicate()
            with open(log_file, "w") as file:
                file.write(stdout.decode())
                print(f'Results have been saved to {log_file}')

metric_list = ['MacroP', 'MacroR', 'MacroF']

init_metric_dic = {metric: [] for metric in metric_list}
lines = ""

all_dic = {}

for dataset, _ in datasets.items():
    all_dic[dataset] = {}
    for method, _ in methods.items():
        print(f'{dataset} {method}')
        this_metric_dic = {metric: [] for metric in metric_list}
        
        lines = ""
        for round in range(rounds):
            temp_metric_dic = process_file(log_file=f"{log_folder}/multi_{dataset}_{method}_{round}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])

        this_metric_avg_dic = {key: sum(value) / rounds for key, value in this_metric_dic.items()}
        line = dic2line(this_metric_avg_dic)
        lines += method + ' & ' + line
        print(lines)

        all_dic[dataset][method] = this_metric_dic
