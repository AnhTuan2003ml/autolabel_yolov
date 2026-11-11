import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def polygon_points(segment):
    points = segment.tolist()
    return [float(value) for pair in points for value in pair]


def infer(source, output, mode, model_name, confidence, device, label_map_path=None):
    if mode not in {'bbox', 'segmentation'}:
        raise ValueError('mode must be bbox or segmentation')
    model = YOLO(model_name)
    label_map = json.loads(Path(label_map_path).read_text(encoding='utf-8')) if label_map_path else {}
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    image_total = len([path for path in Path(source).glob('*') if path.is_file()]) if Path(source).is_dir() else 0
    image_index = 0
    results = model.predict(source=str(source), conf=confidence, device=device, stream=True, verbose=False)
    for result in results:
        image_index += 1
        image_path = Path(result.path)
        detections = []
        names = result.names
        if mode == 'bbox':
            for box in result.boxes:
                points = box.xyxy[0].detach().cpu().tolist()
                name = names[int(box.cls[0])]
                detections.append({'type': 'rectangle', 'label': label_map.get(name, name), 'confidence': float(box.conf[0]), 'points': points})
        elif result.masks is not None:
            for index, polygon in enumerate(result.masks.xy):
                name = names[int(result.boxes.cls[index])]
                detections.append({'type': 'polygon', 'label': label_map.get(name, name), 'confidence': float(result.boxes.conf[index]), 'points': polygon_points(polygon)})
        (output_path / f'{image_path.stem}.json').write_text(json.dumps(detections, ensure_ascii=False), encoding='utf-8')
        print(f'[frame {image_index}/{image_total}] {image_path.name}: {len(detections)} detections', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['bbox', 'segmentation'], default='bbox')
    parser.add_argument('--model', default='yolo11x.pt')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--device', default='0')
    parser.add_argument('--label-map')
    args = parser.parse_args()
    infer(args.source, args.output, args.mode, args.model, args.confidence, args.device, args.label_map)


if __name__ == '__main__':
    main()
