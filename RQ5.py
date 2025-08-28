import os
import time
import subprocess
import re
from scipy.stats import ttest_rel
from copy import deepcopy
import matplotlib.pyplot as plt
from RQutils import *

datasets = {
    'Dataset-M': 'Maldonado_data/',
    'Dataset-VG': 'VG_data/',
}

settings = [
    'FRoM',       # Full model
    '-CCUS',      # without Under-Sampling
    '-MTL',       # without Multi-Task Learning
    '-CCUS-MTL',  # without CCUS and MTL
    '-CCUS+RUS',  # change CCUS to RUS
]

rounds = 10
metric_list = ['F4C1', 'F4C2', 'F4C3']
log_folder = f'logs/RQ4'
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
dic = {'Dataset-M':{}, 'Dataset-VG':{}}
lines = ""

for dataset, path in datasets.items():
    for setting in settings:
        this_metric_dic = deepcopy(init_metric_dic)
        for round in range(rounds):
            temp_metric_dic = process_file(log_file=f"{log_folder}/{dataset}_{setting}_{round}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])
        
        this_metric_avg_dic = {key: sum(value)/rounds for key, value in this_metric_dic.items()}
        dic[dataset][setting] = this_metric_dic
        
        line = dic2line(this_metric_avg_dic, 3)
        lines += setting + line    

print(lines)

p_values = []
dataset = 'Dataset-VG'
s1 = 'FRoM'
s2 = '-CCUS-MTL'
for key, value in dic[dataset][s1].items():
    print(sum(dic[dataset][s1][key])/10, sum(dic[dataset][s2][key])/10)
    t_statistic, p_value = ttest_rel(dic[dataset][s1][key], dic[dataset][s2][key])
    p_values.append(p_value)

for idx, p_value in enumerate(p_values):
    print(f"评估指标 {idx+1}: p-value = {p_value}")

box_colors = ['#AED6F1', '#F9E79F', '#A3E4D7', '#F5B7B1']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 行 3 列：上 row-0 dataset-M, 下 row-1 dataset-VG

global_min = float('inf')
global_max = float('-inf')

for dataset in datasets:
    for metric in metric_list:
        for setting in settings:
            if setting == '-CCUS+RUS':
                continue
            values = dic[dataset][setting][metric]
            global_min = min(global_min, min(values))
            global_max = max(global_max, max(values))

ymin, ymax = global_min * 0.98, global_max * 1.02
# -------------------------------------------

for col_idx, metric in enumerate(metric_list):
    for row_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, col_idx]
        data_to_plot = []
        labels = []
        for setting in settings:
            if setting == '-CCUS+RUS':
                continue
            data_to_plot.append(dic[dataset][setting][metric])
            labels.append(setting)
        
        bp = ax.boxplot(
            data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            medianprops=dict(color='#B22222', linewidth=2.5),
            flierprops=dict(marker='o', markersize=8, markerfacecolor='white',
                            markeredgewidth=2, markeredgecolor='black')
        )

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)

        ax.set_ylim(ymin, ymax)

        if metric == 'F4C1':
            ax.set_title(f'F1 of design debt on {dataset}', fontsize=16, fontweight='bold')
        if metric == 'F4C2':
            ax.set_title(f'F1 of implementation debt on {dataset}', fontsize=16, fontweight='bold')
        if metric == 'F4C3':
            ax.set_title(f'F1 of defect debt on {dataset}', fontsize=16, fontweight='bold')

        ax.set_ylabel('F1', fontsize=16, fontweight='bold')
        ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.tick_params(axis='both', which='both', labelsize=12, width=2)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

plt.tight_layout()
plt.show() 
