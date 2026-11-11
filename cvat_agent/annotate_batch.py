import json
import sys
from pathlib import Path

try:
    from .cdp import evaluate
except ImportError:
    from cdp import evaluate


def fetch_json(url):
    return evaluate(f"fetch({json.dumps(url)}).then(r=>r.json())")


def unlock_and_show_labels():
    return evaluate("[...document.querySelectorAll('.cvat-label-item-button-lock-enabled')].forEach(x=>x.click());[...document.querySelectorAll('.cvat-label-item-button-hidden-enabled')].forEach(x=>x.click());'labels-opened'")


def apply(job_id, prediction_dir, organization):
    unlock_and_show_labels()
    files = sorted(Path(prediction_dir).glob('frame_*.json'))
    predictions = {int(path.stem.split('_')[1]): json.loads(path.read_text(encoding='utf-8')) for path in files}
    existing = fetch_json(f'/api/jobs/{job_id}/annotations?org={organization}')
    labels = {}
    page = 1
    while True:
        data = fetch_json(f'/api/labels?job_id={job_id}&page={page}')
        labels.update({label['name']: label['id'] for label in data.get('results', [])})
        if not data.get('next'):
            break
        page += 1
    frames = set(predictions)
    shapes = [{key: value for key, value in shape.items() if key != 'id'} for shape in existing.get('shapes', []) if shape.get('frame') not in frames]
    added = 0
    for frame, items in predictions.items():
        for item in items:
            label_id = labels.get(item['label'])
            shape_type = item.get('type')
            points = item.get('points', [])
            if label_id is None or shape_type not in {'rectangle', 'polygon'}:
                continue
            if shape_type == 'rectangle' and len(points) != 4:
                continue
            if shape_type == 'polygon' and len(points) < 6:
                continue
            if shape_type == 'polyline' and len(points) < 4:
                continue
            shapes.append({'label_id': label_id, 'frame': frame, 'type': shape_type, 'points': points, 'outside': False, 'occluded': False, 'z_order': 0, 'rotation': 0, 'attributes': []})
            added += 1
    tags = [{key: value for key, value in tag.items() if key != 'id'} for tag in existing.get('tags', []) if tag.get('frame') not in frames]
    payload = {'version': existing.get('version', 0), 'tags': tags, 'shapes': shapes, 'tracks': existing.get('tracks', [])}
    expression = f"fetch('/api/jobs/{job_id}/annotations?org={organization}',{{method:'PUT',credentials:'include',headers:{{'Content-Type':'application/json','X-CSRFToken':decodeURIComponent(document.cookie.split('; ').find(x=>x.startsWith('csrftoken='))?.split('=')[1]||'')}},body:JSON.stringify({json.dumps(payload,separators=(',', ':'))})}}).then(async r=>{{if(!r.ok) throw new Error('HTTP '+r.status+' '+await r.text()); return r.status}})"
    status = evaluate(expression)
    unlock_and_show_labels()
    evaluate("[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Save')?.click(); 'saved'")
    return {'frames': len(frames), 'existing_shapes': len(existing.get('shapes', [])), 'added_shapes': added, 'status': status}


if __name__ == '__main__':
    print(json.dumps(apply(int(sys.argv[1]), sys.argv[2], sys.argv[3]), ensure_ascii=False))
