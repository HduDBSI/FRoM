from RQutils import *
from scipy.stats import ttest_rel

datasets = {
    'dataset-M': 'Maldonado_data/',
    'dataset-VG': 'VG_data/',
    
}

methods = {
    'CNN-based': 'CNN_based.py',
    'XGB-based': 'XGB_based.py',
    'SCGRU': 'scgru/SCGRU.py',
    'FRoM': 'FRoM.py',
}

runs = 10

metric_list = ['P', 'R', 'F']

all_dic = {'bin': {'dataset-VG': {}, 'dataset-M': {}}, 'mul': {'dataset-VG': {}, 'dataset-M': {}}}

def ImpAndp(metric_name, metric_list1, metric_list2):
    avg1 = sum(metric_list1) / runs
    avg2 = sum(metric_list2) / runs

    Imp = round(avg2,4) - round(avg1,4)
    _, p_value = ttest_rel(metric_list1, metric_list2)

    sign = '\\textcolor{myg}{$+$' if Imp > 0 else '\\textcolor{myr}{$-$'
    Imp = Imp if Imp > 0 else -Imp

    p_value = f'\\textbf{{{p_value:.1E}}}' if p_value < 0.05 else f'{p_value:.1E}'
    print(f'{metric_name:3}: {avg2:.3f} ({sign}{Imp:.3f}}}, {p_value})')

for dataset, _ in datasets.items():
    for method, _ in methods.items():
        print(f'{dataset} {method}')
        
        this_metric_dic = {metric: [] for metric in metric_list}
        for run in range(runs):
            temp_metric_dic = process_file(log_file=f"logs/RQ1/binary_{dataset}_{method}_{run}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])
        all_dic['bin'][dataset][method] = this_metric_dic

        this_metric_dic = {metric: [] for metric in metric_list}
        for run in range(runs):
            temp_metric_dic = process_file(log_file=f"logs/RQ2/multi_{dataset}_{method}_{run}.txt", metric_list=metric_list)
            for key, value in temp_metric_dic.items():
                this_metric_dic[key].append(temp_metric_dic[key])
        all_dic['mul'][dataset][method] = this_metric_dic

        for metric in metric_list:
            ImpAndp(metric, all_dic['bin'][dataset][method][metric], all_dic['mul'][dataset][method][metric])

