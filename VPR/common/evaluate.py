"""Evaluate models on photographs taken in separate sessions (TestDB)."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from collections import defaultdict
import time
from VPR.common.data import ROOT, TEST_ROOT, scan_images, edge_key, photo_path
from VPR.common.recognize import VisualDBSearch


def evaluate_model(model_id, batch_size=4, device=None, top_k=5):
    if batch_size < 1 or top_k < 1:
        raise ValueError('batch_size and top_k must be >= 1')
    tests = scan_images(TEST_ROOT, check_map=True)
    predictor = VisualDBSearch(model_id, device=device)
    reference_paths = {r['image_path'].replace('\\', '/').lower() for r in predictor.index}
    for row in tests:
        if row['image_path'].replace('\\', '/').lower() in reference_paths:
            raise RuntimeError('Reference/TestDB leakage: same file path appears in both DBs')
    report = []
    start = time.perf_counter()
    counts = defaultdict(lambda: [0, 0, 0])
    for offset in range(0, len(tests), batch_size):
        sub = tests[offset:offset + batch_size]
        feats = predictor.extractor.batch([photo_path(x) for x in sub])
        for row, descriptor in zip(sub, feats):
            preds = predictor.search_descriptor(descriptor, top_k=top_k)
            correct = edge_key(row)
            predicted = [(p['node_id'], p['approach_from']) for p in preds]
            t1 = int(correct == predicted[0])
            tk = int(correct in predicted)
            key = f"{row['approach_from']}->{row['node_id']}"
            counts[key][0] += 1
            counts[key][1] += t1
            counts[key][2] += tk
            report.append({
                'image': row['image_path'], 'true_edge': key,
                'predicted_edge': preds[0]['edge'],
                'top1_correct': t1, 'topk_correct': tk,
                'top1_cosine': preds[0]['similarity'],
                'topk_edges': '|'.join(p['edge'] for p in preds),
            })
        print(f'[TEST] {model_id}: {min(offset+batch_size, len(tests))}/{len(tests)}', flush=True)
    elapsed = time.perf_counter() - start
    n = len(report)
    out = ROOT / 'TestDB' / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'{model_id}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader(); writer.writerows(report)
    t1 = sum(r['top1_correct'] for r in report)
    tk = sum(r['topk_correct'] for r in report)
    macro1 = sum(good/total for total, good, _ in counts.values()) / len(counts)
    macrok = sum(good/total for total, _, good in counts.values()) / len(counts)
    summary = {
        'model_id': model_id, 'queries': n, 'test_edges': len(counts),
        'top1_micro': round(t1/n, 5), f'top{top_k}_micro': round(tk/n, 5),
        'top1_macro': round(macro1, 5), f'top{top_k}_macro': round(macrok, 5),
        'total_elapsed_seconds': round(elapsed, 3),
        'mean_ms_per_image_including_search': round(elapsed*1000/n, 2),
        'evaluation_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'note': 'Image nearest-neighbor + edge max cosine; no graph/IMU, scores are not probabilities.'
    }
    (out / f'{model_id}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[RESULT] {model_id}: Top1 {t1/n:.2%}, Top{top_k} {tk/n:.2%}, Macro Top1 {macro1:.2%} -> {out}")
    return summary
