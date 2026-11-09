"""Node-VPR CLI: common build/search/evaluate with model selection.

From project root:
    .\\.venv\\Scripts\\python.exe VPR\\cli.py list
    .\\.venv\\Scripts\\python.exe VPR\\cli.py build --models cosplace_resnet18_512
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Support both `python VPR/cli.py ...` and `python -m VPR.cli ...`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VPR.models.registry import MODEL_IDS, MODEL_DIMS, load_model
from VPR.common.data import prepare_folders
from VPR.common.storage import is_complete, load_db


def main(argv=None):
    parser = argparse.ArgumentParser(description='Node-VPR modular model manager')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='Show supported models and DB readiness')
    commands.add_parser('prepare-folders', help='Create incoming-direction folders from KAU_Node_Map.json')
    builder = commands.add_parser('build', help='Build Reference descriptors for 1+ models')
    builder.add_argument('--models', nargs='+', required=True, choices=MODEL_IDS)
    builder.add_argument('--batch-size', type=int, default=8)
    builder.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    builder.add_argument('--force', action='store_true', help='Allow replacing existing DB (normally skipped)')
    recognition = commands.add_parser('recognize', help='Search one query image')
    recognition.add_argument('--model', choices=MODEL_IDS, required=True)
    recognition.add_argument('--image', type=Path, required=True)
    recognition.add_argument('--top-k', type=int, default=5)
    recognition.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    evaluate = commands.add_parser('evaluate', help='Evaluate all photographs in TestDB/images/')
    evaluate.add_argument('--models', nargs='+', choices=MODEL_IDS, default=None)
    evaluate.add_argument('--batch-size', type=int, default=4)
    evaluate.add_argument('--top-k', type=int, default=5)
    evaluate.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    smoke = commands.add_parser('smoke', help='Load model and encode one generated test picture')
    smoke.add_argument('--model', choices=MODEL_IDS, required=True)
    smoke.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    args = parser.parse_args(argv)
    if args.command == 'list':
        for name in MODEL_IDS:
            status = 'READY' if is_complete(name) else 'NOT BUILT'
            print(f'{name:32s} {MODEL_DIMS[name]:5d}-D  {status}')
        return 0
    if args.command == 'prepare-folders':
        prepare_folders()
        return 0
    if args.command == 'build':
        from VPR.common.build_database import build_database
        for name in args.models:
            build_database(name, batch_size=args.batch_size, device=args.device, force=args.force)
        return 0
    if args.command == 'recognize':
        from VPR.common.recognize import VisualDBSearch
        finder = VisualDBSearch(args.model, device=args.device)
        print(f'\n[QUERY] {args.image}')
        for i, item in enumerate(finder.search(args.image, top_k=args.top_k), 1):
            print(f"{i}. {item['node_id']} from_{item['approach_from']}  "
                  f"cosine={item['similarity']:.4f}  image={item['matched_image']}")
        print('[NOTE] cosine similarity is not a probability')
        return 0
    if args.command == 'evaluate':
        from VPR.common.evaluate import evaluate_model
        models = args.models or [name for name in MODEL_IDS if is_complete(name)]
        if not models:
            raise RuntimeError('No descriptor DBs exist. Run build first.')
        for name in models:
            evaluate_model(name, batch_size=args.batch_size, device=args.device, top_k=args.top_k)
        return 0
    if args.command == 'smoke':
        import numpy as np
        model = load_model(args.model, device=args.device)
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(320, dtype=np.uint16)[None, :] * 255 // 319
        image[:, :, 1] = (np.arange(320, dtype=np.uint16)[:, None] * 255 // 319)
        vec = model.descriptor(image)
        print(f'[OK] {args.model}: shape={vec.shape}, L2={np.linalg.norm(vec):.6f}')
        return 0
    raise AssertionError('unreachable')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
