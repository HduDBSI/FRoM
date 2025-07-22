import torch
from transformers import AutoModel
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1, class_num: int = 1):
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
    
encoder = AutoModel.from_pretrained('../roberta/')
clf = Classifier(input_dim=768, dropout=0.1, class_num=4)

content = torch.load('../models/PMFRoM.pth', map_location='cpu')
encoder.load_state_dict(content['encoder'])
clf.load_state_dict(content['clf'])

class CombinedModel(torch.nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.clf = classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embed = outputs.last_hidden_state[:, 0, :]
        logits = self.clf(embed)
        return logits

combined = CombinedModel(encoder, clf)
combined.eval()

dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.int64)
dummy_attention_mask = torch.ones((1, 128), dtype=torch.int64)

torch.onnx.export(
    combined,
    (dummy_input_ids, dummy_attention_mask),
    "../models/PMFRoM.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14
)
