# multi classification
import os
import time
import subprocess
from RQutils import *

train_test = {
    'Maldonado_data/': 'VG_data/',
    'VG_data/': 'Maldonado_data/',
}

methods = {
    'CNN-based': 'CNN_based.py',
    'XGB-based': 'XGB_based.py',
    'SCGRU': 'SCGRU.py',
    'FRoM': 'FRoM.py'
}

log_folder = 'logs/RQ6'

rounds = 10
class_num = 4
conda_env = 'pyten'
device = 'cuda:1'

def getTVT(train_folder, test_folder):
    folder = 'data/' + train_folder
    train_file = 'data/' + train_folder + 'preprocessed/train.jsonl'
    valid_file = 'data/' + test_folder + 'preprocessed/valid.jsonl'
    test_file = 'data/' + test_folder + 'preprocessed/test.jsonl'

    return f'--folder {folder} --train_file {train_file} --valid_file {valid_file} --test_file {test_file}'


for train_folder, test_folder in train_test.items():
    for method, pyfile in methods.items():
        for round in range(rounds):
            os.makedirs(log_folder, exist_ok=True)
            log_file = f"{log_folder}/{train_folder[0]}_{test_folder[0]}_{method}_{round}.txt"
            if os.path.exists(log_file):
                print(f"{log_file} already exists, skipping...")
                continue
            t_round = time.time()
            tvt = getTVT(train_folder, test_folder)

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

for train_folder, test_folder in train_test.items():
    all_dic[train_folder] = {}
    for method, _ in methods.items():
        print(f'{train_folder} {method}')
        this_metric_dic = {metric: [] for metric in metric_list}
        
        lines = ""
        for round in range(rounds):
            temp_metric_dic = process_file(log_file=f"{log_folder}/{train_folder[0]}_{test_folder[0]}_{method}_{round}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])

        this_metric_avg_dic = {key: sum(value) / rounds for key, value in this_metric_dic.items()}
        line = dic2line(this_metric_avg_dic)
        lines += method + ' & ' + line
        print(lines)

        all_dic[train_folder][method] = this_metric_dic
