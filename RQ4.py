from RQutils import *
from scipy.stats import ttest_rel
import time
import subprocess

def ImpAndp(metric_name, metric_list1, metric_list2):
    avg1 = sum(metric_list1) / runs
    avg2 = sum(metric_list2) / runs

    Imp = round(avg2,4) - round(avg1,4)
    _, p_value = ttest_rel(metric_list1, metric_list2)

    sign = '\\textcolor{myg}{$+$' if Imp > 0 else '\\textcolor{myr}{$-$'
    Imp = Imp if Imp > 0 else -Imp

    p_value = f'\\textbf{{{p_value:.1E}}}' if p_value < 0.05 else f'{p_value:.1E}'
    print(f'{metric_name:3}: {avg2:.3f} ({sign}{Imp:.3f}}}, {p_value})')

metric_list = ['MacroP', 'MacroR', 'MacroF']


datasets = {
    'dataset-M': 'Maldonado_data/',
    'dataset-VG': 'VG_data/',
}

methods = {
    'CNN-based': 'CNN_based-2steps.py',
    'XGB-based': 'XGB_based-2steps.py',
    'SCGRU': 'SCGRU-2steps.py',
    'FRoM': 'FRoM-2steps.py'
}

log_folder = 'logs/RQ4'

runs = 10
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
        for run in range(runs):
            os.makedirs(log_folder, exist_ok=True)
            log_file = f"{log_folder}/{dataset}_{method}-2steps_{run}.txt"
            if os.path.exists(log_file):
                print(f"{log_file} already exists, skipping...")
                continue
            t_round = time.time()
            tvt = getTVT(path)

            if pyfile == 'XGB_based-2steps.py':
                command = f"conda run -n {conda_env} python {pyfile} {tvt} --seed {run}"
            else:
                command = f"conda run -n {conda_env} python {pyfile} {tvt} --seed {run} --device {device}"

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, sterr = process.communicate()
            with open(log_file, "w") as file:
                file.write(stdout.decode())
                print(f'Results have been saved to {log_file}')

all_dic = {'one-step': {'dataset-VG': {}, 'dataset-M': {}}, 'two-step': {'dataset-VG': {}, 'dataset-M': {}}}
for method, _ in methods.items():
    for dataset, _ in datasets.items():
    
        print(f'{dataset} {method}')
        
        this_metric_dic = {metric: [] for metric in metric_list}
        for run in range(runs):
            temp_metric_dic = process_file(log_file=f"{log_folder}/{dataset}_{method}-2steps_{run}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])
        all_dic['two-step'][dataset][method] = this_metric_dic

        this_metric_dic = {metric: [] for metric in metric_list}
        for run in range(runs):
            temp_metric_dic = process_file(log_file=f"logs/RQ2/multi_{dataset}_{method}_{run}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])
        all_dic['one-step'][dataset][method] = this_metric_dic

        for metric in metric_list:
            ImpAndp(metric, all_dic['one-step'][dataset][method][metric], all_dic['two-step'][dataset][method][metric])
