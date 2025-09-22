import torch.nn as nn
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self, input_dim=768, dropout=0.1, class_num=4):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, class_num),
        )
        self.reset_parameter()

    def reset_parameter(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=0, mode='fan_in', nonlinearity='relu')
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, emd):
        return self.mlp(emd)

class CustomLoss(nn.Module):
    def __init__(self, weight):
        super(CustomLoss, self).__init__()
        self.weight = weight
    
    def forward(self, logit1, logit2, label):
        prob1, prob2 = F.softmax(logit1, dim=1), F.softmax(logit2, dim=1)
        prob1_log, prob2_log = prob1.log(), prob2.log()

        loss1 = F.nll_loss(prob1_log, label)
        loss2 = F.nll_loss(prob2_log, label)
        # lsce = LabelSmoothCrossEntropy(0.05, 4)
        # loss1, loss2 = lsce(prob1_log, label), lsce(prob2_log, label)

        major_loss = (loss1 + loss2) / 2

        kl12 = F.kl_div(prob1_log, prob2, reduction='batchmean')
        kl21 = F.kl_div(prob2_log, prob1, reduction='batchmean')
        minor_loss = (kl12 + kl21) / 2 / logit1.shape[1]

        return major_loss + self.weight * minor_loss
