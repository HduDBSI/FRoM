import os
import re

def extract_metric(content: str, metric_name: str):
    match = re.search(rf"\b{metric_name}:\s*([0-9]+\.?[0-9]*)", content)
    if match:
        return float(match.group(1))
    return None

def process_file(log_file: str, metric_list: list):
    if not os.path.exists(log_file):
        return None
    with open(log_file, "r") as file:
        content = file.read()
    return {metric: extract_metric(content, metric) for metric in metric_list}

def dic2line(dic: dict, digit: int=3):
    line = ""
    for key, value in dic.items():
        if digit == 3:
            line += "& {:.3f} ".format(value)
        else:
            line += "& {:.4f} ".format(value)
    return ' ' + line + '\\\\ \n'