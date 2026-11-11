import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from .annotate_batch import apply
    from .cdp import ensure_browser, evaluate
    from .job_data import save_chunk
    from .label_agent import LabelDescriptionAgent
except ImportError:
    from annotate_batch import apply
    from cdp import ensure_browser, evaluate
    from job_data import save_chunk
    from label_agent import LabelDescriptionAgent


ROOT = Path(__file__).resolve().parent.parent


def fetch_json(url):
    return evaluate(f"fetch({json.dumps(url)}).then(r=>{{if(!r.ok) throw new Error('HTTP '+r.status); return r.json()}})")


def parse_job_url(url):
    match = re.search(r'/tasks/(\d+)/jobs/(\d+)', url)
    if not match:
        raise ValueError('URL must contain /tasks/<task>/jobs/<job>')
    return int(match.group(1)), int(match.group(2))


def labels_for_job(job_id, destination):
    labels = []
    page = 1
    while True:
        data = fetch_json(f'/api/labels?job_id={job_id}&page={page}')
        labels.extend(item['name'] for item in data.get('results', []))
        if not data.get('next'):
            break
        page += 1
    labels = list(dict.fromkeys(labels))
    destination.write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
    return labels


def collect_images(job_id, start_frame, count, destination):
    images = destination / 'images'
    images.mkdir(parents=True, exist_ok=True)
    chunks = destination / 'chunks'
    frame = start_frame
    chunk = 0
    while frame < start_frame + count:
        save_chunk(job_id, chunks, chunk)
        source = chunks / f'job_{job_id}_chunk_{chunk}'
        for image in sorted(source.glob('*.jpeg')):
            shutil.copy2(image, images / f'frame_{frame:04d}.jpeg')
            frame += 1
            if frame >= start_frame + count:
                break
        chunk += 1
    if frame != start_frame + count:
        raise RuntimeError(f'Expected {count} frames, collected {frame - start_frame}')
    return images


def model_name(mode):
    return 'yolo11x-seg.pt' if mode == 'segmentation' else 'yolo11x.pt'


def run_yolo(images, predictions, mode, label_map):
    predictions.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'yolo_agent.infer', '--source', str(images), '--output', str(predictions), '--mode', mode, '--model', model_name(mode), '--label-map', str(label_map)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError('YOLO inference failed')


def prediction_summary(predictions, labels):
    found = set()
    shapes = 0
    for path in Path(predictions).glob('*.json'):
        for item in json.loads(path.read_text(encoding='utf-8')):
            found.add(item.get('label'))
            shapes += 1
    return {'files': len(list(Path(predictions).glob('*.json'))), 'shapes': shapes, 'found_labels': len(found), 'missing_labels': [label for label in labels if label not in found]}


def build_label_map(labels, destination):
    available = set(labels)
    candidates = {
        'person': ['person', 'pedestrian'],
        'car': ['car'],
        'truck': ['truck'],
        'bus': ['bus'],
        'train': ['train'],
        'motorcycle': ['motorcycle'],
        'bicycle': ['bicycle'],
        'traffic light': ['traffic light', 'traffic_light'],
    }
    mapping = {source: next((target for target in targets if target in available), source) for source, targets in candidates.items()}
    destination.write_text(json.dumps(mapping, ensure_ascii=False), encoding='utf-8')
    return mapping


def main(url, organization, model, mode, output_root, max_points=80, prompt_file=None):
    task_id, job_id = parse_job_url(url)
    ensure_browser(url)
    evaluate(f'location.href={json.dumps(url)}')
    time.sleep(2)
    print('[agent 1/8] Đọc URL job và kết nối Chrome CDP', flush=True)
    job = fetch_json(f'/api/jobs/{job_id}')
    meta = fetch_json(f'/api/jobs/{job_id}/data/meta')
    print(f'[agent 2/8] Tải metadata task={task_id}, job={job_id}', flush=True)
    run_dir = Path(output_root) / f'job_{job_id}'
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    labels_path = run_dir / 'labels.json'
    labels = labels_for_job(job_id, labels_path)
    print(f'[agent 3/8] Lấy toàn bộ {len(labels)} label từ CVAT API', flush=True)
    prompts_path = run_dir / 'label_descriptions.json'
    descriptions = LabelDescriptionAgent().build(labels, prompts_path)
    if prompt_file:
        descriptions = json.loads(Path(prompt_file).read_text(encoding='utf-8'))
        prompts_path.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding='utf-8')
        labels = [item['label'] for item in descriptions]
    print(f'[agent 4/8] Tạo {len(labels)} mô tả label', flush=True)
    images = collect_images(job_id, int(meta['start_frame']), int(job['frame_count']), run_dir)
    print(f'[agent 5/8] Tải {int(job["frame_count"])} frame', flush=True)
    predictions = run_dir / 'predictions'
    label_map = run_dir / 'yolo_label_map.json'
    build_label_map(labels, label_map)
    print(f'[agent 6/8] Chạy YOLO {model_name(mode)} mode={mode}', flush=True)
    run_yolo(images, predictions, mode, label_map)
    summary = prediction_summary(predictions, labels)
    print(f'[agent 7/8] Hậu kiểm {summary["shapes"]} shape, {summary["found_labels"]}/{len(labels)} label', flush=True)
    print('[agent 8/8] Mở khóa, hiển thị label, xóa annotation cũ và upload vào CVAT', flush=True)
    uploaded = apply(job_id, predictions, organization)
    return {'task_id': task_id, 'job_id': job_id, 'frames': int(job['frame_count']), 'model': model_name(mode), 'mode': mode, **summary, **uploaded}


if __name__ == '__main__':
    result = main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 80, sys.argv[7] if len(sys.argv) > 7 else None)
    print(json.dumps(result, ensure_ascii=False), flush=True)
