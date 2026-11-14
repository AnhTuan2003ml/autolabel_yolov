import math
import re


COCO17_NAMES = (
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
)

VISIBILITY_THRESHOLD = 0.3


class SkeletonSchemaError(ValueError):
    pass


def normalize_name(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


COCO17_BY_NAME = {normalize_name(name): index for index, name in enumerate(COCO17_NAMES)}


def numeric_index(name):
    normalized = normalize_name(name)
    match = re.fullmatch(r'(?:numeric_?|keypoint_?)?(\d+)', normalized)
    if not match:
        return None
    number = int(match.group(1))
    if 1 <= number <= len(COCO17_NAMES):
        return number - 1
    return None


def sublabel_mapping(sublabels):
    if not isinstance(sublabels, list) or len(sublabels) != len(COCO17_NAMES):
        raise SkeletonSchemaError('Skeleton must define exactly 17 COCO sublabels')
    mapping = {}
    numeric = []
    for item in sublabels:
        if not isinstance(item, dict):
            raise SkeletonSchemaError('Sublabel metadata must be an object')
        if not isinstance(item.get('id'), int):
            raise SkeletonSchemaError('Sublabel metadata must include an integer id')
        index = numeric_index(item.get('name'))
        if index is not None:
            numeric.append((index, item['id']))
    if numeric:
        if len(numeric) != len(sublabels):
            raise SkeletonSchemaError('Sublabel names mix numeric and COCO names')
        seen = set()
        for index, label_id in numeric:
            if index in seen:
                raise SkeletonSchemaError('Duplicate numeric skeleton sublabel')
            seen.add(index)
            mapping[index] = label_id
    else:
        for item in sublabels:
            index = COCO17_BY_NAME.get(normalize_name(item.get('name')))
            if index is None:
                raise SkeletonSchemaError(f'Unsupported skeleton sublabel: {item.get("name")}')
            if index in mapping:
                raise SkeletonSchemaError(f'Duplicate skeleton sublabel: {item.get("name")}')
            mapping[index] = item['id']
    if set(mapping) != set(range(len(COCO17_NAMES))):
        raise SkeletonSchemaError('Skeleton does not cover all 17 COCO keypoints')
    if len(set(mapping.values())) != len(COCO17_NAMES):
        raise SkeletonSchemaError('Skeleton sublabel ids must be unique')
    return [mapping[index] for index in range(len(COCO17_NAMES))]


def is_compatible(sublabels):
    try:
        sublabel_mapping(sublabels)
    except SkeletonSchemaError:
        return False
    return True


def keypoint_triples(item):
    points = item.get('keypoints') if isinstance(item, dict) else None
    if not isinstance(points, (list, tuple)) or len(points) != len(COCO17_NAMES):
        raise SkeletonSchemaError('Skeleton prediction must contain 17 keypoints')
    triples = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            raise SkeletonSchemaError('Skeleton keypoints must be x, y, confidence triples')
        try:
            triple = [float(point[0]), float(point[1]), float(point[2])]
        except (TypeError, ValueError) as error:
            raise SkeletonSchemaError('Skeleton keypoints must be numeric') from error
        if not all(math.isfinite(value) for value in triple):
            raise SkeletonSchemaError('Skeleton keypoints must be finite')
        triples.append(triple)
    return triples


def skeleton_shape(label_id, frame, item, sublabels):
    mapping = sublabel_mapping(sublabels)
    elements = []
    for index, (x, y, confidence) in enumerate(keypoint_triples(item)):
        elements.append({
            'label_id': mapping[index],
            'type': 'points',
            'points': [x, y],
            'outside': confidence < VISIBILITY_THRESHOLD,
            'occluded': False,
            'z_order': 0,
            'rotation': 0,
            'attributes': [],
        })
    return {
        'label_id': label_id,
        'frame': frame,
        'type': 'skeleton',
        'points': [],
        'outside': False,
        'occluded': False,
        'z_order': 0,
        'rotation': 0,
        'attributes': [],
        'elements': elements,
    }
