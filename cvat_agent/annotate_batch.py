import copy
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlencode

try:
    from .cdp import evaluate
    from .pose_skeleton import SkeletonSchemaError, skeleton_shape
except ImportError:
    from cdp import evaluate
    from pose_skeleton import SkeletonSchemaError, skeleton_shape


def fetch_json(url):
    return evaluate(f"fetch({json.dumps(url)},{{credentials:'include'}}).then(async r=>{{if(!r.ok) throw new Error('HTTP '+r.status+' '+await r.text()); return await r.json().catch(()=>({{}}))}})")


def normalize_label_name(value):
    return ' '.join(str(value).strip().lower().replace('_', ' ').split())


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


def load_predictions(prediction_dir):
    predictions = {}
    for path in sorted(Path(prediction_dir).glob('frame_*.json')):
        try:
            frame = int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            predictions[frame] = data
    return predictions


def allowed_labels(prediction_dir):
    path = Path(prediction_dir).parent / 'allowed_labels.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return None
    return [str(item) for item in data if isinstance(item, (str, int, float))]


def prediction_labels(predictions):
    labels = []
    seen = set()
    for frame in sorted(predictions):
        for item in predictions[frame]:
            label = item.get('label') if isinstance(item, dict) else None
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def finite_points(points):
    if not isinstance(points, (list, tuple)):
        return None
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


def standalone_shape(label_id, frame, shape_type, points):
    return {
        'label_id': label_id,
        'frame': frame,
        'type': shape_type,
        'points': points,
        'outside': False,
        'occluded': False,
        'z_order': 0,
        'rotation': 0,
        'attributes': [],
    }


def track_shape(frame, points, outside):
    return {
        'frame': frame,
        'type': 'rectangle',
        'points': points,
        'outside': outside,
        'occluded': False,
        'z_order': 0,
        'rotation': 0,
        'attributes': [],
    }


def track_objects(label_id, observations, start_frame=None, stop_frame=None):
    frames = sorted(observations)
    shapes = []
    for index, frame in enumerate(frames):
        points = observations[frame]
        shapes.append(track_shape(frame, points, False))
        if index + 1 < len(frames):
            for missing in range(frame + 1, frames[index + 1]):
                shapes.append(track_shape(missing, points, True))
        else:
            closing = frame + 1
            if stop_frame is None or closing <= stop_frame:
                if start_frame is None or closing >= start_frame:
                    shapes.append(track_shape(closing, points, True))
    return {
        'label_id': label_id,
        'frame': frames[0],
        'group': 0,
        'source': 'manual',
        'attributes': [],
        'shapes': shapes,
    }


def build_objects(predictions, records, selected_names=None, start_frame=None, stop_frame=None):
    by_name = {}
    for record in records:
        name = record.get('name')
        if name:
            by_name.setdefault(normalize_label_name(name), record)
    if selected_names is None:
        selected_names = prediction_labels(predictions)
    permitted = {normalize_label_name(name) for name in selected_names}
    selected_ids = set()
    for name in selected_names:
        record = by_name.get(normalize_label_name(name))
        if record is not None and isinstance(record.get('id'), int):
            selected_ids.add(record['id'])
    shapes = []
    groups = {}
    for frame in sorted(predictions):
        for item in predictions[frame]:
            if not isinstance(item, dict):
                continue
            label_name = normalize_label_name(item.get('label', ''))
            if label_name not in permitted:
                continue
            record = by_name.get(label_name)
            if record is None or not isinstance(record.get('id'), int):
                continue
            label_id = record['id']
            shape_type = item.get('type')
            if shape_type == 'skeleton':
                if record.get('type') != 'skeleton':
                    raise SkeletonSchemaError('Prediction label is not a CVAT skeleton label')
                shapes.append(skeleton_shape(label_id, frame, item, record.get('sublabels') or []))
            elif shape_type == 'rectangle':
                points = finite_points(item.get('points'))
                if points is None or len(points) != 4:
                    continue
                track_id = item.get('track_id')
                if isinstance(track_id, int) and not isinstance(track_id, bool):
                    groups.setdefault((label_id, track_id), {})[frame] = points
                else:
                    shapes.append(standalone_shape(label_id, frame, 'rectangle', points))
            elif shape_type == 'polygon':
                points = finite_points(item.get('points'))
                if points is None or len(points) < 6 or len(points) % 2:
                    continue
                shapes.append(standalone_shape(label_id, frame, 'polygon', points))
    tracks = [track_objects(label_id, groups[(label_id, track_id)], start_frame, stop_frame) for label_id, track_id in sorted(groups)]
    if not shapes and not tracks:
        raise ValueError('No valid predictions to apply')
    return {'shapes': shapes, 'tracks': tracks, 'selected_ids': selected_ids}


def in_frame_span(frame, start_frame, stop_frame):
    if start_frame is not None and frame < start_frame:
        return False
    if stop_frame is not None and frame > stop_frame:
        return False
    return True


def track_frames(track):
    frames = []
    frame = track.get('frame')
    if isinstance(frame, int) and not isinstance(frame, bool):
        frames.append(frame)
    for shape in track.get('shapes') or []:
        if isinstance(shape, dict):
            shape_frame = shape.get('frame')
            if isinstance(shape_frame, int) and not isinstance(shape_frame, bool):
                frames.append(shape_frame)
    return frames


def delete_refs(existing, selected_ids, start_frame=None, stop_frame=None):
    shapes = []
    for item in existing.get('shapes') or []:
        if item.get('label_id') not in selected_ids or not isinstance(item.get('id'), int):
            continue
        frame = item.get('frame')
        if not isinstance(frame, int) or not in_frame_span(frame, start_frame, stop_frame):
            continue
        shapes.append(copy.deepcopy(item))
    tracks = []
    for item in existing.get('tracks') or []:
        if item.get('label_id') not in selected_ids or not isinstance(item.get('id'), int):
            continue
        frames = track_frames(item)
        if frames:
            if not all(in_frame_span(frame, start_frame, stop_frame) for frame in frames):
                continue
        elif start_frame is not None or stop_frame is not None:
            continue
        tracks.append(copy.deepcopy(item))
    return shapes, tracks


def patch_annotations(job_id, organization, payload):
    action = payload.get('action') if isinstance(payload, dict) else None
    if action not in ('create', 'delete'):
        raise ValueError('Unsupported annotation action')
    query = urlencode({'org': organization, 'action': action})
    body = {key: value for key, value in payload.items() if key != 'action'}
    expression = f"fetch('/api/jobs/{job_id}/annotations?{query}',{{method:'PATCH',credentials:'include',headers:{{'Content-Type':'application/json','X-CSRFToken':decodeURIComponent(document.cookie.split('; ').find(x=>x.startsWith('csrftoken='))?.split('=')[1]||'')}},body:JSON.stringify({json.dumps(body,separators=(',', ':'))})}}).then(async r=>{{if(!r.ok) throw new Error('HTTP '+r.status+' '+await r.text()); return await r.json().catch(()=>({{}}))}})"
    return evaluate(expression)


def apply(job_id, prediction_dir, organization, start_frame=None, stop_frame=None):
    predictions = load_predictions(prediction_dir)
    records = load_label_records(job_id)
    names = allowed_labels(prediction_dir)
    if names is None:
        names = prediction_labels(predictions)
    objects = build_objects(predictions, records, names, start_frame, stop_frame)
    existing = fetch_json(f'/api/jobs/{job_id}/annotations?{urlencode({"org": organization})}')
    delete_shapes, delete_tracks = delete_refs(existing, objects['selected_ids'], start_frame, stop_frame)
    version = existing.get('version', 0)
    created = patch_annotations(job_id, organization, {
        'version': version,
        'tags': [],
        'shapes': objects['shapes'],
        'tracks': objects['tracks'],
        'action': 'create',
    })
    updated_version = created.get('version', version) if isinstance(created, dict) else version
    if delete_shapes or delete_tracks:
        patch_annotations(job_id, organization, {
            'version': updated_version,
            'tags': [],
            'shapes': delete_shapes,
            'tracks': delete_tracks,
            'action': 'delete',
        })
    evaluate("location.reload(); 'reloaded'")
    return {
        'frames': len(predictions),
        'added_shapes': len(objects['shapes']),
        'added_tracks': len(objects['tracks']),
        'deleted_shapes': len(delete_shapes),
        'deleted_tracks': len(delete_tracks),
    }


if __name__ == '__main__':
    first = int(sys.argv[4]) if len(sys.argv) > 4 else None
    last = int(sys.argv[5]) if len(sys.argv) > 5 else None
    print(json.dumps(apply(int(sys.argv[1]), sys.argv[2], sys.argv[3], first, last), ensure_ascii=False))
