SHAPE_RULES = {
    'pedestrian': 'rectangle',
    'rider': 'rectangle',
    'car': 'rectangle',
    'truck': 'rectangle',
    'bus': 'rectangle',
    'train': 'rectangle',
    'motorcycle': 'rectangle',
    'bicycle': 'rectangle',
    'traffic light': 'rectangle',
    'traffic sign': 'rectangle',
    'area/direct': 'polygon',
    'area/alternative': 'polygon',
    'lane/crosswalk': 'polyline',
    'lane/double white': 'polyline',
    'lane/double yellow': 'polyline',
    'lane/double other': 'polyline',
    'lane/road curb': 'polyline',
    'lane/single white': 'polyline',
    'lane/single yellow': 'polyline',
    'lane/single other': 'polyline',
}


def allowed_labels(labels):
    return [label for label in dict.fromkeys(labels) if label in SHAPE_RULES]


def shape_for_label(label):
    return SHAPE_RULES.get(label)
