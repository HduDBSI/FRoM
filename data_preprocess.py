
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
import os
import json

# pre-compile patterns for comments
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
comment_pattern = re.compile(r'//|/\*|\*/|\*')
nonchar_pattern = re.compile(r'[^\w\s.,!?;:\'\"\-\[\]\(\)@]')
space_pattern = re.compile(r'\s{2,}')
hyphen_pattern = re.compile(r'-{2,}')

def format_comment(comment: str):
    # comment = url_pattern.sub('URL', comment)
    comment = comment_pattern.sub(' ', comment)
    comment = nonchar_pattern.sub(' ', comment)
    comment = hyphen_pattern.sub(' ', comment)
    comment = space_pattern.sub(' ', comment)
    return comment.strip().lower()


def print_class_counts_summary(train_df, valid_df, test_df):

    def print_class_counts(df, df_name):
        print(f"{df_name}: {len(df)}", end="")
        label_counts = df['label'].value_counts().to_dict()
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            print(f" , #{label}: {count}", end="")
        print()
    
    print_class_counts(train_df, "Train")
    print_class_counts(valid_df, "Valid")
    print_class_counts(test_df, "Test")
    print_class_counts(pd.concat([train_df, valid_df, test_df]), "Total")


def split_df(df: pd.DataFrame):
    # Shuffle the dataset randomly
    df = df.sample(frac=1, random_state=42)

    # Split the dataset into features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Split the dataset while maintaining the proportion of positive and negative samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

    # Concatenate the split datasets back into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, valid_df, test_df

def export_data(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, folder: str):
    os.makedirs(folder+'/preprocessed', exist_ok=True)
    os.makedirs(folder+'/raw', exist_ok=True)
    
    total_df_raw = pd.concat([train_df, valid_df, test_df]).reset_index(drop=True)

    if 'project' in total_df_raw.columns:
        total_df_raw[['project', 'comment', 'label']].to_json(folder+'/raw/raw_total.jsonl', orient='records', lines=True)
    else:
        total_df_raw[['comment', 'label']].to_json(folder+'/raw/raw_total.jsonl', orient='records', lines=True)

    export_list = ['comment', 'comment_text', 'label']
    train_df[export_list].to_json(folder+'/preprocessed/train.jsonl', orient='records', lines=True)
    valid_df[export_list].to_json(folder+'/preprocessed/valid.jsonl', orient='records', lines=True)
    test_df[export_list].to_json(folder+'/preprocessed/test.jsonl', orient='records', lines=True)

def process(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, folder: str):

    for df in [train_df, valid_df, test_df]:

        df['comment_text'] = df['comment'].apply(format_comment)

    os.makedirs(folder+'/output', exist_ok=True)

    return train_df, valid_df, test_df

def preprocess_Our_data():

    t_start = time.time()
    print("====== Our Data ======")
    folder = 'data/Our_data'

    projects = [
        "antlr4-4.11.0", "dbeaver-22.2.5", "elasticsearch-8.5.2", "exoplayer-2.18.2",
        "fastjson-1.2.83", "flink-1.15.3", "guava-31.1", "jenkins-2.379", "libgdx-1.11.0", 
        "logstash-8.5.2", "mockito-4.9.0", "openrefine-3.6.2", "presto-0.278", 
        "quarkus-2.14.0", "questdb-6.6", "redisson-3.18.1", "rxjava-3.1.5", "tink-1.7.0"
    ]
    
    df_train_list, df_valid_list, df_test_list = [], [], []

    for project in projects:
        df_tmp = pd.read_csv(f'{folder}/raw/{project}_allLevel_comment.csv', encoding='utf-8-sig')
        df_tmp['project'] = project
        df_tmp = df_tmp[['project', 'comment', 'label']].drop_duplicates(subset=['comment'])

        label_counts = df_tmp['label'].value_counts()
        
        print(project)
        for i in range(2):
            count = label_counts.get(i, 0)
            print(f'{i}: {count}')
        print()

        train_df_tmp, valid_df_tmp, test_df_tmp = split_df(df_tmp)

        df_train_list.append(train_df_tmp)
        df_valid_list.append(valid_df_tmp)
        df_test_list.append(test_df_tmp)

    train_df = pd.concat(df_train_list).reset_index(drop=True)
    valid_df = pd.concat(df_valid_list).reset_index(drop=True)
    test_df = pd.concat(df_test_list).reset_index(drop=True)
    
    print_class_counts_summary(train_df, valid_df, test_df)

    train_df, valid_df, test_df = process(train_df, valid_df, test_df, folder)

    export_data(train_df, valid_df, test_df, folder)
    
    print('Cost Time:', time.time() - t_start)

# https://github.com/Naplues/MAT/
def preprocess_Guo_data():

    t_start = time.time()
    print("====== Guo et al.'s Data ======")
    folder = 'data/Guo_data'

    projects = [
        "Dubbo", "Gradle", "Groovy", "Hive", "Maven", "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"
    ]

    df_train_list, df_valid_list, df_test_list = [], [], []

    for project in projects:

        comments = []
        with open(f'{folder}/raw/data--{project}.txt', 'r') as file:
            for comment in file:
                comments.append(comment)
        
        labels = []
        with open(f'{folder}/raw/label--{project}.txt', 'r') as file:
            for label in file:
                labels.append(1 if label == 'positive\n' else 0)

        df_tmp = pd.DataFrame({'comment': comments, 'label': labels})
        df_tmp['project'] = project

        df_tmp = df_tmp[['project', 'comment', 'label']].drop_duplicates(subset=['comment'])

        label_counts = df_tmp['label'].value_counts()
        print(project)
        for i in range(2):
            count = label_counts.get(i, 0)
            print(f'{i}: {count}')
        print()

        train_df_tmp, valid_df_tmp, test_df_tmp = split_df(df_tmp)

        df_train_list.append(train_df_tmp)
        df_valid_list.append(valid_df_tmp)
        df_test_list.append(test_df_tmp)
    
    train_df = pd.concat(df_train_list).reset_index(drop=True)
    valid_df = pd.concat(df_valid_list).reset_index(drop=True)
    test_df = pd.concat(df_test_list).reset_index(drop=True)

    print_class_counts_summary(train_df, valid_df, test_df)

    train_df, valid_df, test_df = process(train_df, valid_df, test_df, folder)

    export_data(train_df, valid_df, test_df, folder)

    print('Cost Time:', time.time() - t_start)

    
def preprocess_Maldonado_data():

    t_start = time.time()
    print("====== Maldonado et al.'s Data ======")
    folder = 'data/Maldonado_data'

    df = pd.read_csv(folder+'/raw/technical_debt_dataset.csv')

    projects = ['apache-ant-1.7.0', 'argouml',  'columba-1.4-src',
        'emf-2.4.1', 'hibernate-distribution-3.3.2.GA', 'jEdit-4.2',
        'jfreechart-1.0.19', 'apache-jmeter-2.10', 'jruby-1.4.0', 'sql12'
    ]

    df.loc[df['classification'] == 'WITHOUT_CLASSIFICATION', 'classification'] = 0
    df.loc[df['classification'] == 'DESIGN', 'classification'] = 1
    df.loc[df['classification'] == 'IMPLEMENTATION', 'classification'] = 2
    df.loc[df['classification'] == 'DEFECT', 'classification'] = 3
    df = df[df['classification'].isin([0, 1, 2, 3])].reset_index(drop=True)

    df.rename(columns={'projectname': 'project', 'commenttext': 'comment', 'classification': 'label'}, inplace=True)

    df_train_list, df_valid_list, df_test_list = [], [], []

    for project in projects:
        
        df_tmp = df[df['project'] == project].drop_duplicates(subset=['comment'])
        
        label_counts = df_tmp['label'].value_counts()
        
        print(project)
        for i in range(4):
            count = label_counts.get(i, 0)
            print(f'{i}: {count}')
        print()

        train_df_tmp, valid_df_tmp, test_df_tmp = split_df(df_tmp)

        df_train_list.append(train_df_tmp)
        df_valid_list.append(valid_df_tmp)
        df_test_list.append(test_df_tmp)
    
    train_df = pd.concat(df_train_list).reset_index(drop=True)
    valid_df = pd.concat(df_valid_list).reset_index(drop=True)
    test_df = pd.concat(df_test_list).reset_index(drop=True)

    print_class_counts_summary(train_df, valid_df, test_df)

    train_df, valid_df, test_df = process(train_df, valid_df, test_df, folder)

    export_data(train_df, valid_df, test_df, folder)

    print(time.time() - t_start)

# https://doi.org/10.5281/zenodo.4558220
def preprocess_Vidoni_data():

    t_start = time.time()
    print("====== Vidoni's Data ======")
    folder = 'data/Vidoni_data'

    df = pd.read_excel(folder+'/raw/satd-comments-manual-subclass.xlsx', sheet_name="Sheet1")

    df.loc[df['debt'] == 'DESIGN', 'debt'] = 1
    df.loc[df['debt'] == 'REQUIREMENTS', 'debt'] = 2
    df.loc[df['debt'] == 'DEFECT', 'debt'] = 3
    df = df[df['debt'].isin([1, 2, 3])].reset_index(drop=True)

    df.rename(columns={'debt': 'label'}, inplace=True)

    train_df, valid_df, test_df = split_df(df)

    print_class_counts_summary(train_df, valid_df, test_df)

    train_df, valid_df, test_df = process(train_df, valid_df, test_df, folder)

    export_data(train_df, valid_df, test_df, folder)

    print(time.time() - t_start)

def construct_dataset_VG():

    def load_postive_data(jsonl_file):
        comment = []
        comment_text = []
        label = []

        with open(jsonl_file, "r") as file:
            for line in file:
                data = json.loads(line)
                comment.append(data['comment'])
                comment_text.append(data["comment_text"])
                label.append(data["label"])

        return pd.DataFrame({'comment': comment, 'comment_text': comment_text, 'label': label})
    
    def load_negative_data(jsonl_file, n):
        comment = []
        comment_text = []
        label = []

        with open(jsonl_file, "r") as file:
            for line in file:
                data = json.loads(line)
                if data['label'] == 0:
                    comment.append(data['comment'])
                    comment_text.append(data["comment_text"])
                    label.append(data["label"])
        df = pd.DataFrame({'comment': comment, 'comment_text': comment_text, 'label': label})
        random_selected_df = df.sample(n)

        return random_selected_df

    train_df_V = load_postive_data('data/Vidoni_data/preprocessed/train.jsonl')
    valid_df_V = load_postive_data('data/Vidoni_data/preprocessed/valid.jsonl')
    test_df_V = load_postive_data('data/Vidoni_data/preprocessed/test.jsonl')

    train_set_G = load_negative_data('data/Guo_data/preprocessed/train.jsonl', len(train_df_V))
    valid_set_G = load_negative_data('data/Guo_data/preprocessed/valid.jsonl', len(valid_df_V))
    test_set_G = load_negative_data('data/Guo_data/preprocessed/test.jsonl', len(test_df_V))

    train_df = pd.concat([train_df_V, train_set_G], axis=0)
    valid_df = pd.concat([valid_df_V, valid_set_G], axis=0)
    test_df = pd.concat([test_df_V, test_set_G], axis=0)

    export_data(train_df, valid_df, test_df, 'data/VG_data')

# preprocess_Our_data()
# preprocess_Guo_data()
# preprocess_Maldonado_data()
preprocess_Vidoni_data()
# construct_dataset_VG()