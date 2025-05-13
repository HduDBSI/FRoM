from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# calculate the precision, recall, and F1 score for a target class
def PRF4TgtCls(y_true: np.array, y_pred: np.array, tgt_class: int=1, neg_class: int=-1):
    # create copies of true and predicted values
    y_true_copy = np.copy(y_true)
    y_pred_copy = np.copy(y_pred)
    
    # set values not equal to the target class to neg_class
    y_true_copy[y_true_copy != tgt_class] = neg_class
    y_pred_copy[y_pred_copy != tgt_class] = neg_class

    # calculate true positives, false positives, and false negatives
    tp = np.sum((y_true_copy == tgt_class) & (y_pred_copy == tgt_class))
    fp = np.sum((y_true_copy == neg_class) & (y_pred_copy == tgt_class))
    fn = np.sum((y_true_copy == tgt_class) & (y_pred_copy == neg_class))

    # calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def cal_metrics(y_true, y_pred, y_pred_logit, output: bool = False):
    
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred_logit, np.ndarray):
        y_pred_logit = np.array(y_pred_logit)
    metrics = {}

    if len(set(y_true)) == 2:
        metrics['P'], metrics['R'], metrics['F'] = PRF4TgtCls(y_true, y_pred, 1)
        metrics['AUC'] = roc_auc_score(y_true, y_pred_logit)
        
    else:
        bin_y_true = np.where(y_true != 0, 1, 0)
        bin_y_pred = np.where(y_pred != 0, 1, 0)

        metrics['P'], metrics['R'], metrics['F'] = PRF4TgtCls(bin_y_true, bin_y_pred, 1)
        metrics['AUC'] = roc_auc_score(bin_y_true, y_pred_logit)

        # calculate precision, recall and f1-score for class 1
        metrics['P4C1'], metrics['R4C1'], metrics['F4C1'] = PRF4TgtCls(y_true, y_pred, 1)

        # calculate precision, recall and f1-score for class 2
        metrics['P4C2'], metrics['R4C2'], metrics['F4C2'] = PRF4TgtCls(y_true, y_pred, 2)

        # calculate precision, recall and f1-score for class 3
        metrics['P4C3'], metrics['R4C3'], metrics['F4C3'] = PRF4TgtCls(y_true, y_pred, 3)

        metrics['MacroP'] = (metrics['P4C1'] + metrics['P4C2'] + metrics['P4C3']) / 3
        metrics['MacroR'] = (metrics['R4C1'] + metrics['R4C2'] + metrics['R4C3']) / 3
        metrics['MacroF'] = (metrics['F4C1'] + metrics['F4C2'] + metrics['F4C3']) / 3

        metrics['FFF'] = 2 * (metrics['F'] * metrics['MacroF']) / (metrics['F'] + metrics['MacroF']) if (metrics['F'] + metrics['MacroF']) > 0 else 0  

    if output:
        for key, value in metrics.items():
            print("{}: {:.4f}".format(key, value))
        print()

    return metrics

def test4macro():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 1, 0])

    P1, R1, F1 = PRF4TgtCls(y_true, y_pred, 0)
    P2, R2, F2 = PRF4TgtCls(y_true, y_pred, 1)
    P3, R3, F3 = PRF4TgtCls(y_true, y_pred, 2)
    
    # calculate macro precision
    macroP1 = (P1 + P2 + P3) / 3

    # calculate macro recall
    macroR1 = (R1 + R2 + R3) / 3

    # calculate macro F1 score
    macroF1 = (F1 + F2 + F3) / 3

    macroP2 = precision_score(y_true, y_pred, average='macro')
    macroR2 = recall_score(y_true, y_pred, average='macro')
    macroF2 = f1_score(y_true, y_pred, average='macro')

    print(macroP1, macroR1, macroF1)
    print(macroP2, macroR2, macroF2)

def test4avg():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1])
    P1, R1, F1 = PRF4TgtCls(y_true, y_pred, 1)
    
    P2 = precision_score(y_true, y_pred)
    R2 = recall_score(y_true, y_pred)
    F2 = f1_score(y_true, y_pred)

    print(P1, R1, F1)
    print(P2, R2, F2)

def plot_loss(train_losses):
    plt.figure(figsize=(10, 5))  
    plt.plot(train_losses, label='Training Loss')  
    plt.title('Training Loss Over Epochs')  
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')  
    plt.grid(True)  
    plt.legend()  
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')  # 保存图像为 loss.png
    plt.close()  # 关闭图像以释放内存
    
# test4macro()
# test4avg()

