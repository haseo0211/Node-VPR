"""Atomic descriptor storage and legacy DB schema support."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
from VPR.common.data import DB_ROOT
from VPR.models.registry import MODEL_DIMS


def save_atomic_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_atomic_numpy(path, data):
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as out:
            np.save(out, data, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def is_complete(model_id):
    folder = DB_ROOT / model_id
    return all((folder / item).is_file() for item in ('features.npy', 'index.json', 'manifest.json'))


def write_db(model_id, matrix, index, rejected, preprocessing):
    folder = DB_ROOT / model_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 2,
        'model_id': model_id,
        'descriptor_dim': int(matrix.shape[1]),
        'image_size': [320, 320],
        'preprocessing': preprocessing,
        'normalization': 'l2',
        'descriptor_dtype': 'float32',
        'image_count': len(index),
        'rejected_photos': int(rejected),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'feature_file': 'features.npy',
        'index_file': 'index.json',
    }
    # Write index/manifest before final feature file. Interrupted builds are
    # detected by completeness/shape checks before any search or evaluation.
    save_atomic_json(folder / 'index.json', index)
    save_atomic_json(folder / 'manifest.json', manifest)
    save_atomic_numpy(folder / 'features.npy', matrix)
    return folder


def _normalize_index_item(item, row):
    if {'node_id', 'approach_from', 'image_path'} <= set(item):
        return {'row': row, 'node_id': str(item['node_id']),
                'approach_from': str(item['approach_from']), 'image_path': str(item['image_path'])}
    # Previous standalone benchmark stored {image, node, from, edge} and used
    # a different descriptor root. Convert if explicitly loaded.
    if {'node', 'from', 'image'} <= set(item):
        return {'row': row, 'node_id': str(item['node']),
                'approach_from': str(item['from']),
                'image_path': str(item['image'])}
    raise ValueError(f'Unknown DB index schema at row {row}: {item}')


def load_db(model_id):
    folder = DB_ROOT / model_id
    if not is_complete(model_id):
        raise FileNotFoundError(f'Build descriptor DB first: {folder}')
    matrix = np.load(folder / 'features.npy', allow_pickle=False)
    items = json.loads((folder / 'index.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    stored_id = manifest.get('model_id', manifest.get('model'))
    if stored_id != model_id:
        raise ValueError(f'Model ID mismatch: {stored_id!r} vs {model_id!r}')
    if not isinstance(items, list):
        raise ValueError('Index file must be a list')
    index = [_normalize_index_item(item, i) for i, item in enumerate(items)]
    expected_dim = MODEL_DIMS[model_id]
    if matrix.ndim != 2 or matrix.shape != (len(index), expected_dim):
        raise ValueError(f'Corrupt shape {matrix.shape}; expected ({len(index)}, {expected_dim})')
    if manifest.get('image_count') != len(index) or not len(index):
        raise ValueError('Manifest/image count mismatch')
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite descriptor values')
    return matrix.astype(np.float32, copy=False), index, manifest
