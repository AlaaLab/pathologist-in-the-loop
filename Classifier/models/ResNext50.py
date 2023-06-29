import torch
import torch.nn as nn

class Myresnext50(nn.Module):
    def __init__(self, my_pretrained_model, num_classes = 23):
        super(Myresnext50, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, num_classes))
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        
        pred = torch.sigmoid(x.reshape(x.shape[0], 1,self.num_classes))
        return pred


class Myresnext50_knowledge(nn.Module):
    def __init__(self, my_pretrained_model, num_classes = 23):
        super(Myresnext50_knowledge, self).__init__()
        self.pretrained = my_pretrained_model
        self.celltype_layers = nn.Sequential(nn.Linear(1000, 100),
                                             nn.ReLU(),
                                             nn.Linear(100, num_classes))

        self.feature_layers = nn.Sequential(nn.Linear(1000, 100),
                                            nn.ReLU(),
                                            nn.Linear(100, 26))

    def forward(self, x):
        x = self.pretrained(x)
        x1 = self.celltype_layers(x)
        x2 = self.feature_layers(x)

        pred_celltype = torch.sigmoid(x1.reshape(x1.shape[0], 1, num_classes))
        pred_features = torch.sigmoid(x2.reshape(x2.shape[0], 1, 26))
        return pred_celltype, pred_features


