from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import xgboost as xgb
from re import sub, compile
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import  PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
import json
import time
import os
from functools import reduce
import argparse
import numpy as np
import random
from CustomMetrics import cal_metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='data/VG_data')
    parser.add_argument('--load_keywords', type=bool, default=True)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    return parser

def set_seed(seed):
    random.seed(seed)  # Randomness for Python
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed for reproducibility
    np.random.seed(seed)  # Randomness for numpy

class StringFormatter():
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.ls = LancasterStemmer()    
    
    def format(self, string):       
        return self.stem(self.clean(string))

    def clean(self, string):
        pattern = compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 删除链接
        string = pattern.sub(' ', string)
        # clean logs
        string = sub(r'\S+-\S+-\S+\s:\s+(\S|\s)+', '', string)
        string = sub(r'Contributor\(s\)(.+\n)+', '', string)
        string = sub(r'\*|/|=', '', string)
        string = sub(r'-', ' ', string)

        string = sub(r"<TABLE .*?>((.|\n)*?)</TABLE>", " ", string)
        string = sub(r"<table .*?>((.|\n)*?)</table>", " ", string)
        # clean html labels
        string = sub('<[^>]*>','', string)

        string = sub(r"[^A-Za-z(),\+!?\'\`]", " ", string)
        string = sub(r"\'s", " \'s", string)  
        string = sub(r"\'ve", " \'ve", string)
        string = sub(r"n\'t", " n\'t", string)
        string = sub(r"\'re", " \'re", string)
        string = sub(r"\'d", " \'d", string)
        string = sub(r"\'ll", " \'ll", string)
        string = sub(r",", " , ", string)
        string = sub(r"!", "  ", string)
        string = sub(r"\(", " ", string)
        string = sub(r"\)", " ", string)
        string = sub(r"\?", " \? ", string)
        string = sub(r"\+", " ", string)
        string = sub(r"\"", " ", string)
        string = sub(r",", " ", string)
        string = sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    
    def stem(self, string):
        words = word_tokenize(string)
        text = []
        for word in words:
            if word == 'ca':
                text.append('can')
            elif word == 'n\'t':
                text.append('not')
            elif word == 'wo':
                text.append('will')
            else:
                text.append(self.wnl.lemmatize(word))
        text = [self.ps.stem(w) for w in text]
        text = [self.ls.stem(w) for w in text]
        text = ' '.join(text)
        text = text.replace('\\ ?', '\\?')
        if text == '':
            text = '""'
        return text

class XGB_based():
    def __init__(self):
        
        self.cv = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 2))
        
    def buildCountVectorizer(self, comments):
        
        self.cv.fit(comments)
        print('Count Vectorizer built')
    
    def fit(self, X_train, y_train, X_valid, y_valid):

        X_train = self.cv.transform(X_train)
        X_valid = self.cv.transform(X_valid)
        print('word transform to vectors')

        self.clf = xgb.XGBClassifier(
            max_depth=6, 
            n_estimators=1000, 
            colsample_bytree=0.8, 
            objective='binary:logistic' if args.class_num == 2 else 'multi:softmax',
            subsample=0.8, 
            nthread=20, 
            learning_rate=0.06,
            random_state=args.seed
        )

        self.clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=30, verbose=False)

    def classify(self, X_test):
        
        X_test = self.cv.transform(X_test)
        y_pred = self.clf.predict(X_test)
        pred_prob = self.clf.predict_proba(X_test)
        y_pred_prob = 1 - pred_prob[:,0]
        return y_pred, y_pred_prob


def cal_chi_word_class(word, documents_set, labels):
    A1, B1, C1, D1 = 0., 0., 0., 0.
    for doc, label in zip(documents_set, labels): 
        if word in doc:
            A1 += label
            B1 += 1 - label
        else:
            C1 += label
            D1 += 1 - label

    A0, B0, C0, D0 = B1, A1, D1, C1
    chi_word_1 = (A1*D1-C1*B1)**2 / ((A1+B1)*(C1+D1))
    chi_word_0 = (A0*D0-C0*B0)**2 / ((A0+B0)*(C0+D0))
    return chi_word_1, chi_word_0

def chi(word_set, documents_set, labels):
    P1 = sum(labels) / len(labels)
    P0 = 1 - P1
    dic = {}
    for word in word_set:
        chi_word_1, chi_word_0 = cal_chi_word_class(word, documents_set, labels)
        chi_word = P0 * chi_word_0 + P1 * chi_word_1
        dic[word] = chi_word
    
    return dic

# documents = [document_1, document_2, ...], document_i = "word_1 word_2 word_3"
# labels = [label_1, label_2, ...], label_i = 0 or 1
def set_keywords(documents:list, labels:list, percentage=0.1):
    documents_set = [set(document.split()) for document in documents]

    word_set = reduce(set.union, documents_set)

    dic = chi(word_set, documents_set, labels)
    sorted_list = sorted(dic.items(), key=lambda x:x[1], reverse=True)
    sorted_chi_word = [x[0] for x in sorted_list]
    
    top_k_words = sorted_chi_word[:int(percentage * len(sorted_chi_word))]
    return set(top_k_words)

def remove_useless_word(keywords, documents):
    
    def process_one(document):  
        return ' '.join(word for word in document.split() if word in keywords)
              
    return [process_one(document) for document in documents]

def load_data(json_file):
    comments = []
    labels = []
    with open(json_file, "r") as file:
        for line in file:
            data = json.loads(line)
            comments.append(data["comment"])
            labels.append(data["label"])
    
    return comments, labels

def load_keywords():
    with open(f'{args.folder}/output/keywords.txt', 'r') as file:
        keywords = [line.replace("\n", "") for line in file]

    return set(keywords)

def create(lst, n):
    length = len(lst)
    for i in range(length, n, 1):
        if i % 2 == 0:
            rand1, rand2 = int(random.random() * length), int(random.random() * length)
            str1 = lst[rand1]
            str2 = lst[rand2]
            str_l1 = str1.split(' ')
            str_l2 = str2.split(' ')
            str_l = str_l1[:int(len(str_l1) / 2)]
            str_l.extend(str_l2[int(len(str_l2) / 2):])
            str_2 = str_l2[:int(len(str_l2) /2)]
            str_2.extend(str_l1[int(len(str_l1) / 2):])
            s1 = ' '.join(str_l).strip()
            s2 = ' '.join(str_2).strip()
            lst.append(s1)
            lst.append(s2)
    return lst


def calc_jd(nonlist, designlist, defectlist, implemntationlist):
    cls0 = len(nonlist)
    cls1 = len(designlist)
    cls2 = len(defectlist)
    all_list = nonlist + designlist + defectlist + implemntationlist

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    X = vectorizer.fit_transform(all_list)

    tfidf = transformer.fit_transform(X).todense()

    # print('start')
    ans = np.array(tfidf)
    (_, dim) = ans.shape

    # calculate Sb
    ans_0 = ans[0:cls0]
    ans_1 = ans[cls0:cls0 + cls1, :]
    ans_2 = ans[cls0 + cls1: cls0 + cls1 + cls2, :]
    ans_3 = ans[cls0 + cls1 + cls2: , :]
    m = np.mean(ans, axis=0)
    m0 = np.mean(ans_0, axis=0)
    m1 = np.mean(ans_1, axis=0)
    m2 = np.mean(ans_2, axis=0)
    m3 = np.mean(ans_3, axis=0)
    m = m.reshape(dim, 1)
    m0 = m0.reshape(dim,1)
    m1 = m1.reshape(dim, 1)
    m2 = m2.reshape(dim, 1)
    m3 = m3.reshape(dim, 1)
    Sb = ((m0-m).dot((m0-m).T) + (m1 - m).dot((m1 - m).T) + (m2 - m).dot((m2 - m).T) + (m3 - m).dot((m3 - m).T)) / 4

    J1 = np.trace(Sb)

    Jd = J1
    return Jd

def jd_create(X:list, y:list, defect_n=1000, implemntation_n=1000):
    nonlist, defectlist, designlist, implementationlist= [], [], [], []
    for index in range(len(y)):
        if X[index] == '':
            continue
        if y[index] == 0:
            nonlist.append(X[index])
        elif y[index] == 1:
            designlist.append(X[index])
        elif y[index] == 2:
            defectlist.append(X[index])
        else:
            implementationlist.append(X[index])
    best_jd = 0.0
    for i in range(50):
        tmpdefectlist = create(defectlist, defect_n)
        tmpimplemntationlist = create(implementationlist, implemntation_n)
        jd = calc_jd(nonlist, designlist, tmpdefectlist, tmpimplemntationlist)
        if (best_jd < jd):
            best_jd = jd
        # print('best_jd:{}'.format(best_jd))
    return nonlist + designlist + defectlist + implementationlist,  [0] * len(nonlist) + [1] * len(designlist) + [2] * len(defectlist) + [3] * len(implementationlist)

t1 = time.time()

parser = get_parser()
args = parser.parse_args()

args_dict = {arg: getattr(args, arg) for arg in vars(args)} 
print(*[f"{k}: {v}" for k, v in args_dict.items()], sep='\n') 
print()

set_seed(args.seed)

X_train, y_train = load_data(args.folder+"/preprocessed/train.jsonl" if args.train_file == None else args.train_file)
X_valid, y_valid = load_data(args.folder+"/preprocessed/valid.jsonl" if args.valid_file == None else args.valid_file)
X_test, y_test = load_data(args.folder+"/preprocessed/test.jsonl" if args.test_file == None else args.test_file)

sf = StringFormatter()

X_train = [sf.format(x) for x in X_train]
X_valid = [sf.format(x) for x in X_valid]
X_test = [sf.format(x) for x in X_test]
print('texts formated')

if args.load_keywords:
    keywords = load_keywords()
else:
    keywords = set_keywords(documents=X_train, labels=y_train, percentage=0.1)

    os.makedirs(args.folder+'/output', exist_ok=True)
    with open(args.folder+'/output/keywords.txt', 'w') as file:
        for keyword in list(keywords):
            file.write("%s\n" % keyword)

X_train = remove_useless_word(keywords, X_train)
X_valid = remove_useless_word(keywords, X_valid)
X_test = remove_useless_word(keywords, X_test)

# data augment
X_train, y_train = jd_create(X_train, y_train)

if args.class_num == 2:
    y_train = [int(y != 0) for y in y_train]
    y_valid = [int(y != 0) for y in y_valid]
    y_test =  [int(y != 0) for y in y_test]

xgb_based = XGB_based()
xgb_based.buildCountVectorizer(X_train+X_valid+X_test)

xgb_based.fit(X_train, y_train, X_valid, y_valid)

y_pred, y_pred_prob = xgb_based.classify(X_test)
cal_metrics(y_test, y_pred, y_pred_prob, True)

print('total time:', time.time()-t1)