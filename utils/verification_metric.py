import numpy as np
import torch
from ignite.metrics import Metric
from sklearn.metrics import roc_auc_score, roc_curve

def cosine_similarity(x, y):
    """Compute cosine similarity between two feature vectors"""
    x_norm = x / torch.norm(x, dim=1, keepdim=True)
    y_norm = y / torch.norm(y, dim=1, keepdim=True)
    return torch.sum(x_norm * y_norm, dim=1)

class VerificationMetrics(Metric):
    def __init__(self):
        super(VerificationMetrics, self).__init__()
        self.count = 0

    def reset(self):
        self.similarities = []
        self.labels = []

    def update(self, output):
        feat1, feat2, label = output
        # Compute cosine similarity
        sim = cosine_similarity(feat1, feat2)
        self.similarities.extend(sim.cpu().numpy())
        self.labels.extend(label.cpu().numpy())

    def compute(self):
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        # Compute metrics
        auc = roc_auc_score(labels, similarities)
        
        # Compute EER
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        
        # Compute accuracy at optimal threshold
        optimal_threshold = eer_threshold
        predictions = (similarities > optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        return {
            'accuracy': accuracy,
            'eer': eer,
            'auc': auc,
            'threshold': optimal_threshold
        }