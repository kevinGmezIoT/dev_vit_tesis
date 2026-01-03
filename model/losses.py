import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # Positives: same identity. 
            # We exclude the anchor itself by checking the distance > 0? 
            # Actually, standard Triplet Loss includes the anchor in the mask, 
            # but we want the hardest positive that is NOT the anchor.
            # For simplicity and to avoid errors, we'll just take the max of all same-identity samples.
            pos_dists = dist[i][mask[i]]
            if pos_dists.numel() > 0:
                dist_ap.append(pos_dists.max().unsqueeze(0))
            else:
                # If no positives (should not happen with K>=1), use 0
                dist_ap.append(torch.tensor([0.0], device=dist.device))

            # Negatives: different identities
            neg_dists = dist[i][mask[i] == 0]
            if neg_dists.numel() > 0:
                dist_an.append(neg_dists.min().unsqueeze(0))
            else:
                # If no negatives in batch, use a large value so it doesn't contribute to loss
                # or just use dist_ap[-1] + margin to make loss 0
                print(f"Warning: No negatives found in batch for anchor with label {targets[i].item()}. This happens when a batch contains only one identity.")
                dist_an.append(dist_ap[-1] + self.margin)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
