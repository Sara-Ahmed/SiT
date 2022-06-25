
import torch
import torch.nn as nn
class SimCLR(nn.Module):

    def __init__(self, simclr_temp):
        super().__init__()

        self.temp = simclr_temp
        self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, features):

        B2, n = features.size()
        B = B2 // 2 # batch size
        
        labels = torch.cat([torch.arange(B) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = nn.functional.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temp
        return logits, labels

    def forward(self, features):

        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        
        return loss
