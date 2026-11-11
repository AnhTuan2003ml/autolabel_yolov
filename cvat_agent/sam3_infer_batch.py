import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .shape_rules import shape_for_label
except ImportError:
    from shape_rules import shape_for_label


def polygon_from_mask(mask, max_points):
    image = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = 1.5
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while max_points > 0 and len(contour) > max_points:
        epsilon *= 1.5
        contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return contour.reshape(-1).astype(float).tolist() if len(contour) >= 3 else []


def polyline_from_mask(mask, max_points):
    image = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) > 2 and np.array_equal(contour[0], contour[-1]):
        contour = contour[:-1]
    if max_points > 0 and len(contour) > max_points:
        step = max(1, len(contour) // max_points)
        contour = contour[::step][:max_points]
    if len(contour) < 2:
        return []
    return contour.reshape(-1).astype(float).tolist()


def prompt_variants(label, prompt):
    variants = [prompt]
    normalized = label.strip().lower().replace('_', ' ')
    aliases = {
        'vegetation': ['vegetation', 'trees and bushes', 'green plants and grass', 'foliage'],
        'sky': ['sky', 'visible sky and clouds'],
        'road': ['road surface', 'paved roadway'],
        'wall': ['wall', 'solid vertical wall'],
        'fence': ['fence', 'railing'],
        'terrain': ['terrain', 'bare ground'],
    }
    for value in aliases.get(normalized, [label.replace('_', ' ').replace('/', ' ')]):
        if value.lower() not in {item.lower() for item in variants}:
            variants.append(value)
    return variants


def infer_with_processor(processor, image_dir, output_dir, first_frame, labels_path, mode, max_points, progress_path=None):
    prompt_requests = [(item['prompt'], item['label']) for item in json.loads(Path(labels_path).read_text(encoding='utf-8'))]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(Path(image_dir).glob('*.jpeg'))
    for index, image_path in enumerate(image_paths):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            state = processor.set_image(Image.open(image_path).convert('RGB'))
            detections = []
            for prompt, label in prompt_requests:
                prompts = prompt_variants(label, prompt)
                label_detections = []
                for current_prompt in prompts:
                    output = processor.set_text_prompt(state=state, prompt=current_prompt)
                    masks = output.get('masks')
                    boxes = output.get('boxes')
                    scores = output.get('scores')
                    if masks is None or boxes is None or scores is None:
                        continue
                    for item_index in range(len(scores)):
                        confidence = float(scores[item_index].item())
                        shape_type = shape_for_label(label)
                        if shape_type is None:
                            continue
                        if shape_type == 'rectangle':
                            points = boxes[item_index].detach().cpu().tolist()
                        elif shape_type == 'polyline':
                            points = polyline_from_mask(masks[item_index].squeeze().detach().cpu().numpy() > 0, max_points)
                        else:
                            points = polygon_from_mask(masks[item_index].squeeze().detach().cpu().numpy() > 0, max_points)
                        if points:
                            label_detections.append({'type': shape_type, 'label': label, 'confidence': confidence, 'points': points})
                    if label_detections:
                        break
                detections.extend(label_detections)
        (output_dir / f'frame_{first_frame + index:04d}.json').write_text(json.dumps(detections), encoding='utf-8')
        if progress_path:
            Path(progress_path).write_text(json.dumps({'current': index + 1, 'total': len(image_paths), 'frame': first_frame + index}), encoding='utf-8')
        print(f'[frame {index + 1}/{len(image_paths)}] {image_path.name}', flush=True)


def infer(model_path, image_dir, output_dir, first_frame, labels_path, mode, max_points):
    model = build_sam3_image_model(checkpoint_path=model_path)
    processor = Sam3Processor(model, confidence_threshold=0.25)
    infer_with_processor(processor, image_dir, output_dir, first_frame, labels_path, mode, max_points)


if __name__ == '__main__':
    infer(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7]) if len(sys.argv) > 7 else 80)
