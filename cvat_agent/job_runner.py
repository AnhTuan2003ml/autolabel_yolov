import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from .cdp import ensure_browser, evaluate, set_origin
    from .job_data import save_chunk
    from .pose_skeleton import is_compatible
except ImportError:
    from cdp import ensure_browser, evaluate, set_origin
    from job_data import save_chunk
    from pose_skeleton import is_compatible


ROOT = Path(__file__).resolve().parent.parent
MODES = ('bbox', 'segmentation', 'tracking', 'skeleton')
SUPPORTED_TYPES = {'rectangle', 'polygon', 'skeleton'}
TRACKER = 'bytetrack.yaml'
CONFIDENCE = 0.25


def fetch_json(url):
    try:
        return evaluate(f"fetch({json.dumps(url)}).then(r=>{{if(!r.ok) throw new Error('HTTP '+r.status); return r.json()}})")
    except RuntimeError as error:
        if 'HTTP 401' in str(error):
            raise RuntimeError('CVAT chưa đăng nhập trong Chrome profile. Hãy đăng nhập tại tab CVAT vừa mở rồi chạy lại job.') from error
        raise


def parse_job_url(url):
    match = re.search(r'/tasks/(\d+)/jobs/(\d+)', url)
    if not match:
        raise ValueError('URL must contain /tasks/<task>/jobs/<job>')
    return int(match.group(1)), int(match.group(2))


def load_label_records(job_id):
    records = []
    page = 1
    while True:
        data = fetch_json(f'/api/labels?job_id={job_id}&page={page}')
        records.extend(item for item in data.get('results', []) if isinstance(item, dict))
        if not data.get('next'):
            break
        page += 1
    return records


def labels_for_job(job_id, destination):
    labels = list(dict.fromkeys(record['name'] for record in load_label_records(job_id) if record.get('name')))
    destination.write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
    return labels


def normalize_name(value):
    return re.sub(r'\s+', ' ', str(value).strip().lower().replace('_', ' '))


def selected_labels_from_file(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict):
        if isinstance(data.get('selected'), list):
            data = data['selected']
        else:
            data = list(data.keys())
    if isinstance(data, str):
        data = [data]
    selected = []
    for item in data:
        if isinstance(item, str):
            label = item
        elif isinstance(item, dict):
            label = item.get('label') or item.get('name')
        else:
            label = None
        if label:
            selected.append(str(label))
    return selected


def select_label_records(records, selected):
    records = [record for record in records if isinstance(record, dict) and record.get('name')]
    if not selected:
        return records
    index = {}
    for record in records:
        index.setdefault(normalize_name(record['name']), record)
    result = []
    seen = set()
    for name in selected:
        record = index.get(normalize_name(name))
        if record is not None and record.get('id') not in seen:
            seen.add(record.get('id'))
            result.append(record)
    if not result:
        raise ValueError('Selected labels do not match CVAT labels')
    return result


def class_label_map(names):
    mapping = {}
    for name in names:
        value = str(name)
        variants = {value, value.lower(), value.replace('_', ' '), value.replace(' ', '_')}
        for variant in variants:
            mapping.setdefault(variant, value)
            mapping.setdefault(normalize_name(variant), value)
    normalized = {normalize_name(name) for name in names}
    if 'person' not in normalized:
        for name in names:
            if normalize_name(name) == 'pedestrian':
                mapping.setdefault('person', name)
                mapping.setdefault('pedestrian', name)
                break
    return mapping


def choose_skeleton_label(records):
    for record in records:
        if record.get('type') == 'skeleton' and is_compatible(record.get('sublabels') or []):
            return record
    raise ValueError('No selected CVAT skeleton label with 17 COCO-compatible sublabels')


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('mode must be bbox, segmentation, tracking or skeleton')
    return mode


def validate_model(model):
    if str(model).lower() != 'yolo':
        raise ValueError('Only YOLO is enabled')
    return 'yolo'


def job_frame_bounds(job, meta):
    start = int(meta.get('start_frame', job.get('start_frame', 0)))
    stop = meta.get('stop_frame', job.get('stop_frame'))
    if stop is None:
        stop = start + int(job['frame_count']) - 1
    return start, int(stop)


def inference_command(images, predictions, mode, label_map_path, allowed_path):
    command = [
        sys.executable,
        '-m',
        'yolo_agent.infer',
        '--source',
        str(images),
        '--output',
        str(predictions),
        '--mode',
        validate_mode(mode),
        '--confidence',
        str(CONFIDENCE),
        '--device',
        'env',
        '--label-map',
        str(label_map_path),
        '--allowed-labels',
        str(allowed_path),
        '--tracker',
        TRACKER,
    ]
    if os.environ.get('YOLO_MODEL'):
        command.extend(['--model', os.environ['YOLO_MODEL']])
    return command


def prediction_summary(predictions, labels):
    found = set()
    files = sorted(Path(predictions).glob('frame_*.json'))
    shapes = 0
    for path in files:
        items = json.loads(path.read_text(encoding='utf-8'))
        for item in items:
            label = item.get('label')
            if label:
                found.add(label)
                shapes += 1
    missing = [label for label in labels if label not in found]
    return {'files': len(files), 'shapes': shapes, 'found_labels': len(found), 'missing_labels': missing}


def collect_images(job_id, start_frame, count, destination):
    images = destination / 'images'
    images.mkdir(parents=True, exist_ok=True)
    chunk_root = destination / 'chunks'
    frame = start_frame
    chunk = 0
    while frame < start_frame + count:
        try:
            save_chunk(job_id, chunk_root, chunk)
        except Exception:
            break
        source = chunk_root / f'job_{job_id}_chunk_{chunk}'
        for image in sorted(source.glob('*.jpeg')):
            shutil.copy2(image, images / f'frame_{frame:04d}.jpeg')
            frame += 1
            if frame >= start_frame + count:
                break
        chunk += 1
    if frame != start_frame + count:
        raise RuntimeError(f'Expected {count} frames, collected {frame - start_frame}')
    return images


def run(command):
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError('Command failed: ' + ' '.join(command))


def normalize_points(points):
    values = []
    for value in points:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        values.append(number)
    return values


def normalize_predictions(predictions):
    for path in Path(predictions).glob('frame_*.json'):
        items = json.loads(path.read_text(encoding='utf-8'))
        normalized = []
        for item in items:
            shape_type = item.get('type')
            if shape_type not in SUPPORTED_TYPES:
                continue
            points = normalize_points(item.get('points', []))
            if points is None:
                continue
            if shape_type == 'rectangle':
                if len(points) == 4:
                    pass
                elif len(points) >= 4 and len(points) % 2 == 0:
                    points = [min(points[::2]), min(points[1::2]), max(points[::2]), max(points[1::2])]
                else:
                    continue
            elif shape_type == 'polygon':
                if len(points) < 6 or len(points) % 2:
                    continue
            elif shape_type == 'skeleton':
                if item.get('keypoints') is None:
                    continue
            item['type'] = shape_type
            item['points'] = points if shape_type != 'skeleton' else []
            track_id = item.get('track_id')
            if isinstance(track_id, int) and not isinstance(track_id, bool):
                item['track_id'] = track_id
            else:
                item.pop('track_id', None)
            normalized.append(item)
        path.write_text(json.dumps(normalized, ensure_ascii=False), encoding='utf-8')


def log(step, message):
    print(f'[agent {step}] {message}', flush=True)


def ensure_ui():
    try:
        urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=0.5).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        subprocess.Popen([sys.executable, '-m', 'cvat_agent.ui', '--open'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(url, organization, model='yolo', mode='segmentation', output_root='runs', max_points=80, prompt_file=None):
    validate_model(model)
    validate_mode(mode)
    ensure_ui()
    log('1/8', 'Đọc URL job và kết nối Chrome CDP')
    task_id, job_id = parse_job_url(url)
    ensure_browser(url)
    set_origin(url)
    evaluate(f'location.href={json.dumps(url)}')
    time.sleep(3)
    log('2/8', f'Tải metadata task={task_id}, job={job_id}')
    job = fetch_json(f'/api/jobs/{job_id}')
    meta = fetch_json(f'/api/jobs/{job_id}/data/meta')
    start_frame, stop_frame = job_frame_bounds(job, meta)
    run_dir = Path(output_root) / f'job_{job_id}'
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    log('3/8', 'Lấy toàn bộ label metadata từ CVAT API theo tất cả trang')
    records = load_label_records(job_id)
    labels = [record['name'] for record in records if record.get('name')]
    (run_dir / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
    (run_dir / 'labels_meta.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    selected = selected_labels_from_file(prompt_file)
    selected_records = select_label_records(records, selected)
    if mode == 'skeleton':
        skeleton = choose_skeleton_label(selected_records)
        selected_records = [skeleton]
        label_map = {'person': skeleton['name'], 'pedestrian': skeleton['name']}
    else:
        label_map = class_label_map([record['name'] for record in selected_records])
    allowed = [record['name'] for record in selected_records]
    label_map_path = run_dir / 'label_map.json'
    allowed_path = run_dir / 'allowed_labels.json'
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False), encoding='utf-8')
    allowed_path.write_text(json.dumps(allowed, ensure_ascii=False), encoding='utf-8')
    log('4/8', 'Lọc ' + ', '.join(allowed))
    images = collect_images(job_id, int(meta['start_frame']), int(job['frame_count']), run_dir)
    log('5/8', f'Tải {int(job["frame_count"])} frame vào {images}')
    predictions = run_dir / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    log('6/8', f'Chạy YOLO mode={mode}')
    run(inference_command(images, predictions, mode, label_map_path, allowed_path))
    normalize_predictions(predictions)
    summary = prediction_summary(predictions, allowed)
    log('7/8', f"Hậu kiểm {summary['shapes']} shape, {summary['found_labels']}/{len(allowed)} label")
    if summary['missing_labels']:
        log('7/8', 'Label chưa có prediction: ' + ', '.join(summary['missing_labels']))
    if summary['shapes'] == 0:
        raise RuntimeError('YOLO produced no shapes; refusing to modify CVAT annotations')
    log('8/8', 'Hiển thị label, xóa annotation cũ của label đã chọn và upload vào CVAT')
    run([sys.executable, '-m', 'cvat_agent.annotate_batch', str(job_id), str(predictions), organization, str(start_frame), str(stop_frame)])
    return {'task_id': task_id, 'job_id': job_id, 'frames': int(job['frame_count']), 'model': 'yolo', 'mode': mode, **summary}


if __name__ == '__main__':
    try:
        result = main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 80, sys.argv[7] if len(sys.argv) > 7 else None)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        print(f'[agent error] {error}', flush=True)
        sys.exit(1)
