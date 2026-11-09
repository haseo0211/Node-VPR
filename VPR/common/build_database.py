"""Model-independent Reference DB builder."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import UnidentifiedImageError
from VPR.common.data import PHOTO_ROOT, DB_ROOT, scan_images, photo_path
from VPR.common.storage import is_complete, load_db, write_db
from VPR.models.registry import load_model


def build_database(model_id: str, batch_size=8, device=None, force=False):
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    out = DB_ROOT / model_id
    if out.exists() and not force:
        # Existing complete reference DB is preserved by default.
        if is_complete(model_id):
            _, index, _ = load_db(model_id)
            print(f'[SKIP] Existing DB preserved: {out} ({len(index)} images). Use --force to rebuild.')
            return len(index)
        raise FileExistsError(f'Partial DB directory already exists: {out}. Inspect or use --force.')
    records = scan_images(PHOTO_ROOT, check_map=True)
    extractor = load_model(model_id, device=device)
    vectors = []
    valid = []
    rejected = 0
    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        prepared_images = []
        good = []
        for record in batch_records:
            path = photo_path(record)
            try:
                prepared_images.append(extractor.get_rgb(path))
                good.append(record)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                print(f'[WARN] Cannot load {path}: {exc}')
                rejected += 1
        if good:
            chunk = extractor.batch(prepared_images)
            for record, vector in zip(good, chunk):
                valid.append({**record, 'row': len(valid)})
                vectors.append(vector)
        if start == 0 or (start // batch_size) % 12 == 0 or start + batch_size >= len(records):
            print(f'[DB] {model_id}: {len(valid)} / {len(records)} (rejected={rejected})', flush=True)
    if not vectors:
        raise RuntimeError('No successfully encoded images')
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    folder = write_db(model_id, matrix, valid, rejected, extractor.interpolation)
    print(f'[OK] {model_id}: {matrix.shape} -> {folder}')
    return len(valid)
