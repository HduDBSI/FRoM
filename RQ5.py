import os
import time
import subprocess
import re
from scipy.stats import ttest_rel
from copy import deepcopy
from RQutils import *

datasets = {
    'dataset-M': 'Maldonado_data/',
    'dataset-VG': 'VG_data/',
}

settings = [
    'FRoM',       # Full model
    '-CCUS',      # without Under-Sampling
    '-MTL',       # without Multi-Task Learning
    '-CCUS-MTL',  # without CCUS and MTL
    '-CCUS+RUS',  # change CCUS to RUS
]

rounds = 10
metric_list = ['MacroP', 'MacroR', 'MacroF']
log_folder = f'logs/RQ5'
conda_env = 'pyten'

argument = {
    'device': 'cuda:0',
    'weight': 0.4,
    'class_num': 4,
    'threshold': 0.5,
    'model_name': 'FRoM.pth'
}

pyfile = 'FRoM.py'
def update_argument(argument:dict, path:str) -> dict:
    argument['train_file'] = 'data/' + path + 'preprocessed/train.jsonl'
    argument['valid_file'] = 'data/' + path + 'preprocessed/valid.jsonl'
    argument['test_file'] = 'data/' + path + 'preprocessed/test.jsonl'
    argument['folder'] = 'data/' + path

    return argument

for dataset, path in datasets.items():
    argument = update_argument(argument, path)
    for round in range(rounds):
        for setting in settings:
            os.makedirs(log_folder, exist_ok=True)
            log_file = f"{log_folder}/{dataset}_{setting}_{round}.txt"
                
            if os.path.exists(log_file):
                print(f"{log_file} already exists, skipping...")
                continue
            
            argument['seed'] = round

            if setting == 'FRoM':
                argument['class_balance'] = 'CCUS'
                argument['MTL'] = 1
            elif setting == '-CCUS':
                argument['class_balance'] = 'None'
                argument['MTL'] = 1
            elif setting == '-CCUS+RUS':
                argument['class_balance'] = 'RUS'
                argument['MTL'] = 1
            elif setting == '-MTL':
                argument['class_balance'] = 'CCUS'
                argument['MTL'] = 0
            elif setting == '-CCUS-MTL':
                argument['class_balance'] = 'None'
                argument['MTL'] = 0

            argument_str = ' '.join([f'--{key} {value}' for key, value in argument.items()])

            command = f"conda run -n {conda_env} python {pyfile} {argument_str}"
                         
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, sterr = process.communicate()
            with open(log_file, "w") as file:
                file.write(stdout.decode())
                print(f'Results have been saved to {log_file}')


init_metric_dic = {metric: [] for metric in metric_list}
dic = {}
lines = ""

for dataset, path in datasets.items():
    for setting in settings:
        this_metric_dic = deepcopy(init_metric_dic)
        for round in range(rounds):
            temp_metric_dic = process_file(log_file=f"{log_folder}/{dataset}_{setting}_{round}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])
        
        this_metric_avg_dic = {key: sum(value)/rounds for key, value in this_metric_dic.items()}
        dic[setting] = this_metric_dic
        
        line = dic2line(this_metric_avg_dic, 4)
        lines += setting + line    

print(lines)

p_values = []

s1 = 'FRoM'
s2 = '-CCUS'
for key, value in dic[s1].items():
    print(sum(dic[s1][key])/10, sum(dic[s2][key])/10)
    t_statistic, p_value = ttest_rel(dic[s1][key], dic[s2][key])
    p_values.append(p_value)

# 输出每个评估指标的p值
for idx, p_value in enumerate(p_values):
    print(f"评估指标 {idx+1}: p-value = {p_value}")
