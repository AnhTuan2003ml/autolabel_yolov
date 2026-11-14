import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yolo_agent import infer as infer_module


class FakeBox:
    def __init__(self, cls=0, conf=0.9, xyxy=(0.0, 0.0, 10.0, 10.0), track_id=None):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [list(xyxy)]
        self.id = None if track_id is None else [track_id]


class FakeMasks:
    def __init__(self, polygons):
        self.xy = polygons


class FakeKeypoints:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, path, names, boxes=None, masks=None, keypoints=None):
        self.path = str(path)
        self.names = names
        self.boxes = boxes if boxes is not None else []
        self.masks = masks
        self.keypoints = keypoints


class FakeModel:
    def __init__(self, predict_results=None, track_results=None):
        self.predict_results = predict_results if predict_results is not None else []
        self.track_results = list(track_results or [])
        self.predict_calls = []
        self.track_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return self.predict_results

    def track(self, **kwargs):
        self.track_calls.append(kwargs)
        return [self.track_results.pop(0)]


def make_images(root, names):
    for name in names:
        (Path(root) / name).write_bytes(b'')


def read_predictions(root, stem):
    return json.loads((Path(root) / f'{stem}.json').read_text(encoding='utf-8'))


class TrackingTests(unittest.TestCase):
    def test_one_model_chronological_frames_and_persistent_ids(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0002.jpg', 'frame_0001.jpg'])
            model = FakeModel(track_results=[
                FakeResult(Path(source) / 'frame_0001.jpg', ['person'], [FakeBox(track_id=7)]),
                FakeResult(Path(source) / 'frame_0002.jpg', ['person'], [FakeBox(track_id=7)]),
            ])
            with patch.object(infer_module, 'load_model', return_value=model) as loader:
                infer_module.infer(source, output, 'tracking', None, 0.4, 'cpu')
            loader.assert_called_once_with('yolo11n.pt')
            self.assertEqual(len(model.track_calls), 2)
            self.assertIn('frame_0001.jpg', model.track_calls[0]['source'])
            self.assertIn('frame_0002.jpg', model.track_calls[1]['source'])
            self.assertTrue(all(call['persist'] is True for call in model.track_calls))
            self.assertTrue(all(call['tracker'] == 'bytetrack.yaml' for call in model.track_calls))
            for name in ('frame_0001', 'frame_0002'):
                detection = read_predictions(output, name)[0]
                self.assertEqual(detection['type'], 'rectangle')
                self.assertEqual(detection['track_id'], 7)
                self.assertIsInstance(detection['track_id'], int)

    def test_missing_box_id_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0001.jpg'])
            model = FakeModel(track_results=[
                FakeResult(Path(source) / 'frame_0001.jpg', ['person'], [FakeBox(track_id=4), FakeBox(track_id=None)]),
            ])
            with patch.object(infer_module, 'load_model', return_value=model):
                infer_module.infer(source, output, 'tracking', 'yolo11n.pt', 0.25, 'cpu')
            detections = read_predictions(output, 'frame_0001')
            self.assertEqual(detections[0]['track_id'], 4)
            self.assertNotIn('track_id', detections[1])
            self.assertEqual(detections[1]['type'], 'rectangle')


class SkeletonTests(unittest.TestCase):
    def test_pose_result_keeps_seventeen_coco_triples(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0001.jpg'])
            points = [[float(index), float(index + 1), 0.75] for index in range(17)]
            model = FakeModel(predict_results=[
                FakeResult(Path(source) / 'frame_0001.jpg', ['person'], [FakeBox(cls=0, conf=0.8)], keypoints=FakeKeypoints([points])),
            ])
            with patch.object(infer_module, 'load_model', return_value=model) as loader:
                infer_module.infer(source, output, 'skeleton', None, 0.25, 'cpu')
            loader.assert_called_once_with('yolo11n-pose.pt')
            detection = read_predictions(output, 'frame_0001')[0]
            self.assertEqual(detection['type'], 'skeleton')
            self.assertEqual(detection['label'], 'person')
            self.assertEqual(detection['points'], [])
            self.assertEqual(len(detection['keypoints']), 17)
            self.assertTrue(all(len(point) == 3 for point in detection['keypoints']))
            self.assertEqual(detection['keypoints'][0], [0.0, 1.0, 0.75])

    def test_missing_or_incomplete_keypoints_are_skipped(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0001.jpg', 'frame_0002.jpg'])
            model = FakeModel(predict_results=[
                FakeResult(Path(source) / 'frame_0001.jpg', ['person'], [FakeBox()], keypoints=None),
                FakeResult(Path(source) / 'frame_0002.jpg', ['person'], [FakeBox()], keypoints=FakeKeypoints([[[1.0, 2.0, 0.9]] * 16])),
            ])
            with patch.object(infer_module, 'load_model', return_value=model):
                infer_module.infer(source, output, 'skeleton', 'yolo11n-pose.pt', 0.25, 'cpu')
            self.assertEqual(read_predictions(output, 'frame_0001'), [])
            self.assertEqual(read_predictions(output, 'frame_0002'), [])


class CompatibilityTests(unittest.TestCase):
    def test_bbox_default_and_segmentation_default(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0001.jpg'])
            image_path = Path(source) / 'frame_0001.jpg'
            bbox_result = FakeResult(image_path, ['person'], [FakeBox(cls=0, conf=0.6)])
            del bbox_result.masks
            bbox_model = FakeModel(predict_results=[bbox_result])
            with patch.object(infer_module, 'load_model', return_value=bbox_model) as loader:
                infer_module.infer(source, output, 'bbox', None, 0.25, 'cpu')
            loader.assert_called_once_with('yolo11x.pt')
            bbox = read_predictions(output, 'frame_0001')[0]
            self.assertEqual(bbox['type'], 'rectangle')
            self.assertEqual(bbox['label'], 'person')
            self.assertEqual(bbox['points'], [0.0, 0.0, 10.0, 10.0])
            seg_model = FakeModel(predict_results=[FakeResult(
                image_path,
                ['person'],
                [FakeBox(cls=0, conf=0.7)],
                masks=FakeMasks([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]]),
            )])
            with patch.object(infer_module, 'load_model', return_value=seg_model) as loader:
                infer_module.infer(source, output, 'segmentation', None, 0.25, 'cpu')
            loader.assert_called_once_with('yolo11x-seg.pt')
            segmentation = read_predictions(output, 'frame_0001')[0]
            self.assertEqual(segmentation['type'], 'polygon')
            self.assertEqual(segmentation['label'], 'person')
            self.assertEqual(segmentation['points'], [0.0, 0.0, 10.0, 0.0, 10.0, 10.0])

    def test_allowed_labels_filter_after_label_map_and_model_override(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0001.jpg'])
            label_map = Path(source) / 'labels.json'
            label_map.write_text(json.dumps({'person': 'Human', 'car': 'Vehicle'}), encoding='utf-8')
            allowed = Path(source) / 'allowed.json'
            allowed.write_text(json.dumps(['Human']), encoding='utf-8')
            model = FakeModel(predict_results=[FakeResult(
                Path(source) / 'frame_0001.jpg',
                ['person', 'bicycle', 'car'],
                [FakeBox(cls=0), FakeBox(cls=1), FakeBox(cls=2)],
            )])
            with patch.object(infer_module, 'load_model', return_value=model) as loader:
                infer_module.infer(source, output, 'bbox', 'custom.pt', 0.25, 'cpu', str(label_map), 'bytetrack.yaml', str(allowed))
            loader.assert_called_once_with('custom.pt')
            detections = read_predictions(output, 'frame_0001')
            self.assertEqual([item['label'] for item in detections], ['Human'])

    def test_env_device_uses_ultralytics_auto_selection(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            make_images(source, ['frame_0001.jpg'])
            model = FakeModel(predict_results=[FakeResult(Path(source) / 'frame_0001.jpg', ['person'], [FakeBox()])])
            with patch.object(infer_module, 'load_model', return_value=model):
                infer_module.infer(source, output, 'bbox', None, 0.25, 'env')
            self.assertIsNone(model.predict_calls[0]['device'])
            self.assertIsNone(infer_module.resolve_device('env'))
            self.assertEqual(infer_module.resolve_device('cpu'), 'cpu')


if __name__ == '__main__':
    unittest.main()
