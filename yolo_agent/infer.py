import argparse
import json
import math
from pathlib import Path


MODEL_DEFAULTS = {
    'bbox': 'yolo11x.pt',
    'segmentation': 'yolo11x-seg.pt',
    'tracking': 'yolo11n.pt',
    'skeleton': 'yolo11n-pose.pt',
}

MODES = tuple(MODEL_DEFAULTS)

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def load_model(model_name):
    from ultralytics import YOLO
    return YOLO(model_name)


def resolve_device(device):
    if device is None:
        return None
    value = str(device).strip()
    if value.lower() in ('', 'env'):
        return None
    return value


def scalar(value):
    if value is None:
        return None
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def to_list(value):
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def polygon_points(segment):
    values = to_list(segment)
    if values and not isinstance(values[0], (list, tuple)):
        values = list(zip(values[::2], values[1::2]))
    return [float(value) for pair in values for value in pair]


def box_points(box):
    values = to_list(box.xyxy)
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [float(value) for value in values]


def class_name(names, index):
    if isinstance(names, dict):
        return names.get(index, names.get(str(index), str(index)))
    if 0 <= index < len(names):
        return names[index]
    return str(index)


def mapped_label(names, index, label_map):
    name = class_name(names, index)
    return label_map.get(name, name)


def allowed_labels(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = list(data.keys())
    elif isinstance(data, str):
        data = [data]
    return {str(item) for item in data}


def selected_labels(path):
    data = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    return data if isinstance(data, dict) else {}


def image_paths(source):
    path = Path(source)
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file():
        return [path]
    return []


def result_path(result, fallback):
    value = getattr(result, 'path', None)
    return Path(value) if value else Path(fallback)


def result_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def keypoint_rows(result):
    keypoints = getattr(result, 'keypoints', None)
    data = to_list(getattr(keypoints, 'data', None)) if keypoints is not None else []
    if len(data) == 17 and all(isinstance(row, (list, tuple)) and len(row) >= 3 and not isinstance(row[0], (list, tuple)) for row in data):
        return [data]
    return data


def keypoint_triples(points):
    if not isinstance(points, (list, tuple)) or len(points) != 17:
        return None
    triples = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return None
        try:
            triple = [float(point[0]), float(point[1]), float(point[2])]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in triple):
            return None
        triples.append(triple)
    return triples


def write_detections(output_path, image_path, detections, image_index, image_total):
    (output_path / f'{image_path.stem}.json').write_text(json.dumps(detections, ensure_ascii=False), encoding='utf-8')
    print(f'[frame {image_index}/{image_total}] {image_path.name}: {len(detections)} detections', flush=True)


def infer(source, output, mode, model_name, confidence, device, label_map_path=None, tracker='bytetrack.yaml', allowed_labels_path=None):
    if mode not in MODEL_DEFAULTS:
        raise ValueError('mode must be bbox, segmentation, tracking or skeleton')
    model = load_model(model_name or MODEL_DEFAULTS[mode])
    label_map = selected_labels(label_map_path)
    permitted = allowed_labels(allowed_labels_path)
    device = resolve_device(device)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    if mode == 'tracking':
        images = image_paths(source)
        image_total = len(images)
        for image_index, image_path in enumerate(images, start=1):
            detections = []
            results = result_list(model.track(source=str(image_path), persist=True, tracker=tracker, conf=confidence, device=device, verbose=False))
            for result in results:
                names = result.names
                for index, box in enumerate(result.boxes):
                    name = mapped_label(names, int(scalar(box.cls) or 0), label_map)
                    if permitted is not None and name not in permitted:
                        continue
                    detection = {'type': 'rectangle', 'label': name, 'confidence': float(scalar(box.conf)), 'points': box_points(box)}
                    track_id = scalar(getattr(box, 'id', None))
                    if track_id is not None:
                        detection['track_id'] = int(track_id)
                    detections.append(detection)
            write_detections(output_path, image_path, detections, image_index, image_total)
        return
    image_total = len([path for path in Path(source).glob('*') if path.is_file()]) if Path(source).is_dir() else 0
    results = model.predict(source=str(source), conf=confidence, device=device, stream=True, verbose=False)
    image_index = 0
    for result in results:
        image_index += 1
        image_path = result_path(result, source)
        detections = []
        names = result.names
        masks = getattr(result, 'masks', None)
        polygons = to_list(masks.xy) if masks is not None else []
        rows = keypoint_rows(result) if mode == 'skeleton' else []
        for index, box in enumerate(result.boxes):
            name = mapped_label(names, int(scalar(box.cls) or 0), label_map)
            if permitted is not None and name not in permitted:
                continue
            if mode == 'bbox':
                detections.append({'type': 'rectangle', 'label': name, 'confidence': float(scalar(box.conf)), 'points': box_points(box)})
            elif mode == 'segmentation' and index < len(polygons):
                points = polygon_points(polygons[index])
                if len(points) >= 6:
                    detections.append({'type': 'polygon', 'label': name, 'confidence': float(scalar(box.conf)), 'points': points})
            elif mode == 'skeleton':
                keypoints = keypoint_triples(rows[index]) if index < len(rows) else None
                if keypoints is not None:
                    detections.append({'type': 'skeleton', 'label': name, 'confidence': float(scalar(box.conf)), 'points': [], 'keypoints': keypoints})
        write_detections(output_path, image_path, detections, image_index, image_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=list(MODES), default='bbox')
    parser.add_argument('--model', default=None)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--device', default='0')
    parser.add_argument('--label-map')
    parser.add_argument('--allowed-labels', '--allowed-labels-path', dest='allowed_labels_path')
    parser.add_argument('--tracker', default='bytetrack.yaml')
    args = parser.parse_args()
    infer(args.source, args.output, args.mode, args.model, args.confidence, args.device, args.label_map, args.tracker, args.allowed_labels_path)


if __name__ == '__main__':
    main()
