"""History metrics inspired by SynGA."""
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class MetricsHistory:
    def __init__(self):
        self.memory = OrderedDict()  # smiles -> score
        self.prev_metrics = None
        self.proposals = 0
        self.has_updates = False
        self.num_repeats = 0

    def __len__(self):
        return len(self.memory)

    def commit(self, smiles: str, score: float):
        if smiles in self.memory:
            self.num_repeats += 1
        else:
            self.memory[smiles] = score
            self.has_updates = True

    def top_sorted(self) -> List[Tuple[str, float]]:
        return sorted(self.memory.items(), key=lambda kv: kv[1], reverse=True)

    def metrics(self, budget: int, ks: List[int] = [10, 100]) -> Dict[str, float]:
        sorted_items = self.top_sorted()
        if not sorted_items:
            return {}
        sorted_scores = [s for _, s in sorted_items]
        metrics: Dict[str, float] = {
            "oracle_calls": len(self),
            "proposals": self.proposals,
            "repeats": self.num_repeats,
            "1st": sorted_scores[0],
        }
        for k in ks:
            if len(sorted_scores) < k:
                pad = [sorted_scores[-1]] * (k - len(sorted_scores))
                vals = sorted_scores + pad
            else:
                vals = sorted_scores[:k]
            mean = float(np.mean(vals))
            prev = self.prev_metrics or {}
            prev_mean = prev.get(f"top{k}/mean", mean)
            prev_auc = prev.get(f"top{k}/auc", 0.0)
            prev_calls = prev.get("oracle_calls", 0)
            auc = prev_auc + 0.5 * (mean + prev_mean) * (len(self) - prev_calls)
            metrics[f"top{k}/mean"] = mean
            metrics[f"top{k}/auc"] = (auc + mean * (budget - len(self))) / max(1, budget)
        self.prev_metrics = dict(metrics)
        self.has_updates = False
        return metrics
