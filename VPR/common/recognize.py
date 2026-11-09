"""Model-independent directed-edge search. Cosine similarity is not confidence."""
from __future__ import annotations
import numpy as np
from VPR.common.storage import load_db
from VPR.common.data import edge_key
from VPR.models.registry import load_model


class VisualDBSearch:
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.features, self.index, self.manifest = load_db(model_id)
        self.extractor = load_model(model_id, device=device)
        # Existing initial ResNet18-256 DB used PIL/torchvision bilinear 320x320;
        # the model-specific loader keeps that convention for compatibility.
        self.index_by_edge = {}
        for i, item in enumerate(self.index):
            self.index_by_edge.setdefault(edge_key(item), []).append(i)
        print(f'[DB] Loaded {len(self.index)} descriptors ({len(self.index_by_edge)} edges)')

    def search_descriptor(self, query, top_k=5):
        if top_k < 1:
            raise ValueError('top_k must be >= 1')
        if query.shape != (self.features.shape[1],):
            raise ValueError('Query descriptor dimension mismatch')
        scores = self.features @ query
        results = []
        for (node, approach_from), ids in self.index_by_edge.items():
            best_idx = max(ids, key=lambda i: scores[i])
            results.append({
                'node_id': node,
                'approach_from': approach_from,
                'edge': f'{approach_from}->{node}',
                'similarity': float(scores[best_idx]),
                'matched_image': self.index[best_idx]['image_path'],
            })
        return sorted(results, key=lambda d: d['similarity'], reverse=True)[:top_k]

    def search(self, image, top_k=5):
        return self.search_descriptor(self.extractor.descriptor(image), top_k=top_k)
