"""Node map parsing, directed-edge labels, and image enumeration."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / 'NodeMap' / 'KAU_Node_Map.json'
PHOTO_ROOT = ROOT / 'VisualDB' / 'images'
TEST_ROOT = ROOT / 'TestDB' / 'images'
DB_ROOT = ROOT / 'VisualDB' / 'descriptors'
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def get_map():
    if not MAP.exists():
        return None
    data = json.loads(MAP.read_text(encoding='utf-8-sig'))
    nodes = data.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f'{MAP}: nodes must be a nonempty list')
    labels = {}
    for node in nodes:
        uid = str(node['id'])
        label = str(node.get('node_id') or node.get('label') or '').strip()
        if not label or label in labels.values():
            raise ValueError(f'Duplicate or empty node_id: {label}')
        labels[uid] = label
    return data, labels


def prepare_folders():
    loaded = get_map()
    if loaded is None:
        raise FileNotFoundError(f'Missing map: {MAP}')
    data, uid_to_name = loaded
    folders = set()
    for edge in data.get('edges', []):
        a, b = uid_to_name[str(edge['from'])], uid_to_name[str(edge['to'])]
        folders.add(PHOTO_ROOT / b / f'from_{a}')
        if edge.get('bidirectional', True):
            folders.add(PHOTO_ROOT / a / f'from_{b}')
    seen_entrances = set()
    for e in data.get('entrances', []):
        name = str(e['id'])
        if not re.fullmatch(r'E[1-9]\d*', name) or name in seen_entrances:
            raise ValueError(f'Invalid or repeated entrance ID: {name}')
        seen_entrances.add(name)
        folders.add(PHOTO_ROOT / uid_to_name[str(e['node'])] / f'from_{name}')
    for folder in sorted(folders):
        folder.mkdir(parents=True, exist_ok=True)
    print(f'[OK] Prepared {len(folders)} directional folders (entrances: {len(seen_entrances)})')
    return len(folders)


def scan_images(root: Path, *, check_map: bool = False):
    if not root.is_dir():
        raise FileNotFoundError(f'Missing image directory: {root}')
    allowed = None
    if check_map:
        loaded = get_map()
        if loaded is not None:
            allowed = set(loaded[1].values())
            print(f'[MAP] {len(allowed)} node IDs loaded from {MAP}')
        else:
            print('[MAP] KAU_Node_Map.json not found; node validation skipped')
    records = []
    ignored = 0
    heic = 0
    for path in sorted(root.rglob('*'), key=lambda p: p.as_posix().casefold()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in {'.heic', '.heif'}:
            heic += 1
            continue
        if ext not in VALID_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) != 3 or not rel.parts[1].startswith('from_'):
            ignored += 1
            continue
        node, from_name = rel.parts[0], rel.parts[1][5:]
        if not node or not from_name:
            ignored += 1
            continue
        if allowed is not None and node not in allowed:
            raise ValueError(f'Unknown target node in {rel}; check {MAP.name}')
        records.append({
            'row': len(records),
            'node_id': node,
            'approach_from': from_name,
            'image_path': path.relative_to(ROOT).as_posix() if root.is_relative_to(ROOT) else str(path),
        })
    if ignored:
        print(f'[WARN] {ignored} image(s) outside Node/from_SOURCE folder structure ignored')
    if heic:
        print(f'[WARN] {heic} HEIC/HEIF photos ignored; convert to JPG/PNG')
    if not records:
        raise RuntimeError(f'No valid photos in {root}')
    print(f'[DATA] {len(records)} photos, {len({edge_key(x) for x in records})} directed edges: {root}')
    return records


def edge_key(record):
    return (record['node_id'], record['approach_from'])


def photo_path(record):
    path = Path(record['image_path'])
    return path if path.is_absolute() else ROOT / path
