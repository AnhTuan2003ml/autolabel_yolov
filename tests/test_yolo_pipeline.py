import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvat_agent import annotate_batch, cdp, job_runner, pose_skeleton, ui


COCO_NAMES = pose_skeleton.COCO17_NAMES


def records():
    return [
        {'id': 10, 'name': 'car', 'type': 'rectangle'},
        {'id': 11, 'name': 'person', 'type': 'rectangle'},
        {'id': 12, 'name': 'road', 'type': 'polygon'},
    ]


def skeleton_records(reverse=False, numeric=False):
    names = list(COCO_NAMES)
    if reverse:
        names = list(reversed(names))
    sublabels = []
    for name in names:
        index = COCO_NAMES.index(name)
        sublabels.append({'id': 100 + index, 'name': str(index + 1) if numeric else name})
    return [{'id': 20, 'name': 'pose', 'type': 'skeleton', 'sublabels': sublabels}]


def write_predictions(directory, frame, items):
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / f'frame_{frame:04d}.json').write_text(json.dumps(items), encoding='utf-8')


def keypoints():
    return [[float(index), float(index + 1), 0.9 if index != 1 else 0.1] for index in range(17)]


def existing_shape(shape_id, label_id, frame, shape_type='rectangle'):
    points = [10.0, 20.0, 30.0, 40.0] if shape_type == 'rectangle' else [0.0, 0.0, 10.0, 0.0, 10.0, 10.0]
    return {
        'id': shape_id,
        'type': shape_type,
        'frame': frame,
        'label_id': label_id,
        'group': 0,
        'source': 'manual',
        'occluded': False,
        'outside': False,
        'z_order': 0,
        'rotation': 0.0,
        'points': points,
        'attributes': [],
        'elements': [],
    }


def existing_track_shape(frame, outside=False):
    return {
        'type': 'rectangle',
        'frame': frame,
        'outside': outside,
        'occluded': False,
        'z_order': 0,
        'rotation': 0.0,
        'points': [0.0, 0.0, 10.0, 10.0],
        'attributes': [],
        'source': 'manual',
    }


def existing_track(track_id, label_id, frame, frames):
    return {
        'id': track_id,
        'frame': frame,
        'label_id': label_id,
        'group': 0,
        'source': 'manual',
        'attributes': [],
        'elements': [],
        'shapes': [existing_track_shape(item) for item in frames],
    }


def existing_tag(tag_id, label_id, frame):
    return {
        'id': tag_id,
        'frame': frame,
        'label_id': label_id,
        'group': 0,
        'source': 'manual',
        'attributes': [],
    }


class RunnerTests(unittest.TestCase):
    def test_inference_command_uses_yolo_cli_without_model_override(self):
        with patch.dict(os.environ, {'YOLO_MODEL': ''}):
            command = job_runner.inference_command('images', 'predictions', 'tracking', 'labels.json', 'allowed.json')
        values = ' '.join(command)
        self.assertIn('-m yolo_agent.infer', values)
        self.assertIn('--mode tracking', values)
        self.assertIn('--confidence 0.25', values)
        self.assertIn('--device env', values)
        self.assertIn('--tracker bytetrack.yaml', values)
        self.assertNotIn('sam3', values.lower())
        self.assertNotIn('--model', command)

    def test_inference_command_reads_model_override_from_environment(self):
        with patch.dict(os.environ, {'YOLO_MODEL': 'custom.pt'}):
            command = job_runner.inference_command('images', 'predictions', 'skeleton', 'labels.json', 'allowed.json')
        self.assertEqual(command[-2:], ['--model', 'custom.pt'])

    def test_mode_and_model_validation(self):
        for mode in ('bbox', 'segmentation', 'tracking', 'skeleton'):
            self.assertEqual(job_runner.validate_mode(mode), mode)
        with self.assertRaises(ValueError):
            job_runner.validate_mode('sam3')
        self.assertEqual(job_runner.validate_model('yolo'), 'yolo')
        with self.assertRaises(ValueError):
            job_runner.validate_model('sam3')

    def test_main_dispatches_yolo_cli_and_has_no_sam3_worker(self):
        commands = []

        def fake_run(command):
            commands.append(command)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(job_runner, 'ensure_ui'), \
                patch.object(job_runner, 'ensure_browser'), \
                patch.object(job_runner, 'set_origin'), \
                patch.object(job_runner, 'evaluate'), \
                patch.object(job_runner.time, 'sleep'), \
                patch.object(job_runner, 'fetch_json', side_effect=[{'frame_count': 1}, {'start_frame': 0}]), \
                patch.object(job_runner, 'load_label_records', return_value=[{'id': 1, 'name': 'car', 'type': 'rectangle'}]), \
                patch.object(job_runner, 'collect_images', return_value=Path(root) / 'images'), \
                patch.object(job_runner, 'run', side_effect=fake_run), \
                patch.object(job_runner, 'normalize_predictions'), \
                patch.object(job_runner, 'prediction_summary', return_value={'files': 1, 'shapes': 1, 'found_labels': 1, 'missing_labels': []}):
            result = job_runner.main('https://cvat.example/tasks/1/jobs/2', 'org', 'yolo', 'tracking', root)
        self.assertEqual(result['model'], 'yolo')
        self.assertIn('yolo_agent.infer', commands[0])
        self.assertFalse(hasattr(job_runner, 'ensure_worker'))

    def test_all_four_modes_dispatch_to_yolo_infer(self):
        for mode in ('bbox', 'segmentation', 'tracking', 'skeleton'):
            commands = []

            def fake_run(command, commands=commands):
                commands.append(command)

            with tempfile.TemporaryDirectory() as root, \
                    patch.object(job_runner, 'ensure_ui'), \
                    patch.object(job_runner, 'ensure_browser'), \
                    patch.object(job_runner, 'set_origin'), \
                    patch.object(job_runner, 'evaluate'), \
                    patch.object(job_runner.time, 'sleep'), \
                    patch.object(job_runner, 'fetch_json', side_effect=[{'frame_count': 1}, {'start_frame': 0}]), \
                    patch.object(job_runner, 'load_label_records', return_value=records() + skeleton_records()), \
                    patch.object(job_runner, 'collect_images', return_value=Path(root) / 'images'), \
                    patch.object(job_runner, 'run', side_effect=fake_run), \
                    patch.object(job_runner, 'normalize_predictions'), \
                    patch.object(job_runner, 'prediction_summary', return_value={'files': 1, 'shapes': 1, 'found_labels': 1, 'missing_labels': []}):
                result = job_runner.main('https://cvat.example/tasks/1/jobs/2', 'org', 'yolo', mode, root)
            self.assertEqual(result['mode'], mode)
            self.assertIn('yolo_agent.infer', commands[0])
            self.assertEqual(commands[0][commands[0].index('--mode') + 1], mode)
            self.assertIn('cvat_agent.annotate_batch', commands[1])

    def test_job_frame_bounds_use_metadata_then_frame_count(self):
        self.assertEqual(job_runner.job_frame_bounds({'frame_count': 5}, {'start_frame': 10, 'stop_frame': 14}), (10, 14))
        self.assertEqual(job_runner.job_frame_bounds({'frame_count': 5}, {'start_frame': 10}), (10, 14))
        self.assertEqual(job_runner.job_frame_bounds({'frame_count': 5}, {'start_frame': 10, 'stop_frame': 13}), (10, 13))

    def test_runner_passes_job_frame_bounds_to_annotation_upload(self):
        commands = []

        def fake_run(command):
            commands.append(command)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(job_runner, 'ensure_ui'), \
                patch.object(job_runner, 'ensure_browser'), \
                patch.object(job_runner, 'set_origin'), \
                patch.object(job_runner, 'evaluate'), \
                patch.object(job_runner.time, 'sleep'), \
                patch.object(job_runner, 'fetch_json', side_effect=[{'frame_count': 3, 'start_frame': 10, 'stop_frame': 12}, {'start_frame': 10, 'stop_frame': 12}]), \
                patch.object(job_runner, 'load_label_records', return_value=[{'id': 1, 'name': 'car', 'type': 'rectangle'}]), \
                patch.object(job_runner, 'collect_images', return_value=Path(root) / 'images'), \
                patch.object(job_runner, 'run', side_effect=fake_run), \
                patch.object(job_runner, 'normalize_predictions'), \
                patch.object(job_runner, 'prediction_summary', return_value={'files': 3, 'shapes': 1, 'found_labels': 1, 'missing_labels': []}):
            result = job_runner.main('https://cvat.example/tasks/1/jobs/2', 'org', 'yolo', 'tracking', root)
        self.assertEqual(result['frames'], 3)
        self.assertIn('cvat_agent.annotate_batch', commands[1])
        self.assertEqual(commands[1][-2:], ['10', '12'])

    def test_class_filter_aliases(self):
        pedestrian = job_runner.class_label_map(['pedestrian'])
        self.assertEqual(pedestrian['person'], 'pedestrian')
        both = job_runner.class_label_map(['person', 'pedestrian'])
        self.assertEqual(both['person'], 'person')
        self.assertEqual(both['pedestrian'], 'pedestrian')
        underscored = job_runner.class_label_map(['traffic_light'])
        self.assertEqual(underscored['traffic light'], 'traffic_light')
        self.assertEqual(underscored['traffic_light'], 'traffic_light')

    def test_selected_labels_match_metadata_by_normalized_name(self):
        selected = job_runner.select_label_records([{'id': 1, 'name': 'traffic_light'}], ['traffic light'])
        self.assertEqual(selected[0]['id'], 1)

    def test_normalize_predictions_preserves_explicit_types_and_track_id(self):
        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [
                {'type': 'rectangle', 'label': 'car', 'confidence': 0.9, 'points': [1, 2, 3, 4], 'track_id': 7},
                {'type': 'polygon', 'label': 'road', 'confidence': 0.8, 'points': [0, 0, 4, 0, 4, 4]},
                {'type': 'skeleton', 'label': 'pose', 'confidence': 0.7, 'points': [], 'keypoints': keypoints()},
                {'type': 'tag', 'label': 'car', 'confidence': 0.9, 'points': []},
            ])
            job_runner.normalize_predictions(root)
            items = json.loads((Path(root) / 'frame_0001.json').read_text(encoding='utf-8'))
        self.assertEqual([item['type'] for item in items], ['rectangle', 'polygon', 'skeleton'])
        self.assertEqual(items[0]['track_id'], 7)
        self.assertEqual(items[2]['points'], [])
        self.assertEqual(len(items[2]['keypoints']), 17)


class AnnotationTests(unittest.TestCase):
    def test_native_track_grouping_class_ids_gaps_and_end(self):
        predictions = {
            1: [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 10, 10], 'track_id': 7}],
            3: [{'type': 'rectangle', 'label': 'car', 'points': [1, 1, 11, 11], 'track_id': 7}],
        }
        objects = annotate_batch.build_objects(predictions, records(), ['car'])
        self.assertEqual(objects['selected_ids'], {10})
        self.assertEqual(len(objects['tracks']), 1)
        track = objects['tracks'][0]
        self.assertEqual(track['label_id'], 10)
        self.assertEqual([shape['frame'] for shape in track['shapes']], [1, 2, 3, 4])
        self.assertEqual([shape['outside'] for shape in track['shapes']], [False, True, False, True])

    def test_track_on_final_job_frame_has_no_outside_shape(self):
        predictions = {5: [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 10, 10], 'track_id': 7}]}
        objects = annotate_batch.build_objects(predictions, records(), ['car'], start_frame=5, stop_frame=5)
        shapes = objects['tracks'][0]['shapes']
        self.assertEqual([shape['frame'] for shape in shapes], [5])
        self.assertEqual([shape['outside'] for shape in shapes], [False])

    def test_one_frame_track_closes_only_within_job_bounds(self):
        predictions = {5: [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 10, 10], 'track_id': 7}]}
        final = annotate_batch.build_objects(predictions, records(), ['car'], start_frame=3, stop_frame=5)
        self.assertEqual([shape['frame'] for shape in final['tracks'][0]['shapes']], [5])
        self.assertEqual([shape['outside'] for shape in final['tracks'][0]['shapes']], [False])
        room = annotate_batch.build_objects(predictions, records(), ['car'], start_frame=3, stop_frame=6)
        self.assertEqual([shape['frame'] for shape in room['tracks'][0]['shapes']], [5, 6])
        self.assertEqual([shape['outside'] for shape in room['tracks'][0]['shapes']], [False, True])

    def test_missing_track_id_becomes_standalone_rectangle(self):
        predictions = {2: [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}]}
        objects = annotate_batch.build_objects(predictions, records(), ['car'])
        self.assertEqual(objects['tracks'], [])
        self.assertEqual(objects['shapes'][0]['type'], 'rectangle')
        self.assertEqual(objects['shapes'][0]['label_id'], 10)

    def test_unknown_prediction_label_is_not_fabricated(self):
        predictions = {
            1: [
                {'type': 'rectangle', 'label': 'dog', 'points': [0, 0, 4, 4]},
                {'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]},
            ],
        }
        objects = annotate_batch.build_objects(predictions, records(), ['car', 'dog'])
        self.assertEqual(objects['selected_ids'], {10})
        self.assertEqual(len(objects['shapes']), 1)
        self.assertEqual(objects['shapes'][0]['label_id'], 10)

    def test_skeleton_reordered_sublabel_ids_and_invisible_points(self):
        predictions = {1: [{'type': 'skeleton', 'label': 'pose', 'points': [], 'keypoints': keypoints()}]}
        objects = annotate_batch.build_objects(predictions, skeleton_records(reverse=True), ['pose'])
        shape = objects['shapes'][0]
        self.assertEqual(shape['label_id'], 20)
        self.assertEqual(shape['points'], [])
        self.assertEqual(shape['type'], 'skeleton')
        self.assertEqual(shape['elements'][0]['label_id'], 100)
        self.assertEqual(shape['elements'][1]['label_id'], 101)
        self.assertEqual(shape['elements'][0]['points'], [0.0, 1.0])
        self.assertFalse(shape['elements'][0]['outside'])
        self.assertTrue(shape['elements'][1]['outside'])

    def test_numeric_sublabels_map_by_number_not_array_order(self):
        predictions = {1: [{'type': 'skeleton', 'label': 'pose', 'points': [], 'keypoints': keypoints()}]}
        objects = annotate_batch.build_objects(predictions, skeleton_records(reverse=True, numeric=True), ['pose'])
        self.assertEqual(objects['shapes'][0]['elements'][0]['label_id'], 100)
        self.assertEqual(objects['shapes'][0]['elements'][16]['label_id'], 116)

    def test_unsupported_skeleton_schema_fails_before_mutation(self):
        bad = [{'id': 20, 'name': 'pose', 'type': 'skeleton', 'sublabels': [{'id': 100 + index, 'name': f'x{index}'} for index in range(17)]}]
        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [{'type': 'skeleton', 'label': 'pose', 'points': [], 'keypoints': keypoints()}])
            with patch.object(annotate_batch, 'load_label_records', return_value=bad), \
                    patch.object(annotate_batch, 'fetch_json') as fetcher, \
                    patch.object(annotate_batch, 'patch_annotations') as patcher:
                with self.assertRaises(pose_skeleton.SkeletonSchemaError):
                    annotate_batch.apply(2, root, 'org')
        fetcher.assert_not_called()
        patcher.assert_not_called()

    def test_apply_preserves_unrelated_objects_and_deletes_after_create(self):
        deleted_shape = existing_shape(100, 10, 1)
        kept_shape = existing_shape(101, 11, 1)
        deleted_track = existing_track(200, 10, 1, [1, 2])
        kept_track = existing_track(201, 11, 1, [1])
        existing = {
            'version': 3,
            'tags': [existing_tag(300, 10, 1)],
            'shapes': [deleted_shape, kept_shape],
            'tracks': [deleted_track, kept_track],
        }
        calls = []
        reloads = []

        def fake_patch(job_id, organization, payload):
            calls.append(payload)
            return {'version': 4}

        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json', return_value=existing), \
                    patch.object(annotate_batch, 'patch_annotations', side_effect=fake_patch), \
                    patch.object(annotate_batch, 'evaluate', side_effect=lambda expression: reloads.append(expression)):
                result = annotate_batch.apply(2, root, 'org')
        self.assertEqual([call['action'] for call in calls], ['create', 'delete'])
        self.assertEqual(calls[0]['tracks'], [])
        self.assertEqual(calls[0]['shapes'][0]['label_id'], 10)
        self.assertNotIn('id', calls[0]['shapes'][0])
        self.assertEqual(calls[1]['shapes'], [deleted_shape])
        self.assertEqual(calls[1]['tracks'], [deleted_track])
        self.assertIsNot(calls[1]['shapes'][0], deleted_shape)
        self.assertIsNot(calls[1]['tracks'][0], deleted_track)
        self.assertIsNot(calls[1]['tracks'][0]['shapes'], deleted_track['shapes'])
        self.assertEqual(calls[1]['tags'], [])
        self.assertEqual(result['deleted_shapes'], 1)
        self.assertEqual(len(reloads), 1)

    def test_apply_delete_payload_contains_complete_existing_objects(self):
        deleted_shape = existing_shape(100, 10, 1)
        deleted_track = existing_track(200, 10, 1, [1, 2])
        existing = {
            'version': 7,
            'tags': [],
            'shapes': [deleted_shape],
            'tracks': [deleted_track],
        }
        calls = []

        def fake_patch(job_id, organization, payload):
            calls.append(payload)
            return {'version': 8}

        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json', return_value=existing), \
                    patch.object(annotate_batch, 'patch_annotations', side_effect=fake_patch), \
                    patch.object(annotate_batch, 'evaluate'):
                annotate_batch.apply(2, root, 'org')
        payload_shape = calls[1]['shapes'][0]
        payload_track = calls[1]['tracks'][0]
        self.assertEqual(payload_shape, deleted_shape)
        self.assertEqual(payload_track, deleted_track)
        self.assertEqual(payload_shape['id'], 100)
        self.assertEqual(payload_shape['type'], 'rectangle')
        self.assertEqual(payload_shape['frame'], 1)
        self.assertEqual(payload_shape['points'], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(payload_shape['attributes'], [])
        self.assertEqual(payload_shape['elements'], [])
        self.assertEqual(payload_track['id'], 200)
        self.assertEqual(payload_track['shapes'], deleted_track['shapes'])
        self.assertEqual(payload_track['shapes'][0]['type'], 'rectangle')
        self.assertEqual(payload_track['shapes'][0]['points'], [0.0, 0.0, 10.0, 10.0])
        self.assertNotEqual(set(payload_shape), {'id'})
        self.assertNotEqual(set(payload_track), {'id'})

    def test_failed_create_prevents_delete_and_reload(self):
        calls = []
        reloads = []

        def fake_patch(job_id, organization, payload):
            calls.append(payload)
            raise RuntimeError('create failed')

        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json', return_value={'version': 1, 'shapes': [existing_shape(100, 10, 1)], 'tracks': [], 'tags': []}), \
                    patch.object(annotate_batch, 'patch_annotations', side_effect=fake_patch), \
                    patch.object(annotate_batch, 'evaluate', side_effect=lambda expression: reloads.append(expression)):
                with self.assertRaises(RuntimeError):
                    annotate_batch.apply(2, root, 'org')
        self.assertEqual([call['action'] for call in calls], ['create'])
        self.assertEqual(reloads, [])

    def test_patch_annotations_moves_action_to_encoded_query(self):
        expressions = []
        with patch.object(annotate_batch, 'evaluate', side_effect=lambda expression: expressions.append(expression) or {'version': 2}):
            result = annotate_batch.patch_annotations(2, 'team one', {'version': 1, 'tags': [], 'shapes': [], 'tracks': [], 'action': 'create'})
        self.assertEqual(result, {'version': 2})
        expression = expressions[0]
        self.assertIn('/api/jobs/2/annotations?org=team+one&action=create', expression)
        self.assertIn("method:'PATCH'", expression)
        self.assertNotIn('"action"', expression)
        self.assertNotIn('"action":', expression)

    def test_patch_annotations_rejects_missing_or_unsupported_action_before_evaluate(self):
        with patch.object(annotate_batch, 'evaluate') as evaluator:
            with self.assertRaises(ValueError):
                annotate_batch.patch_annotations(2, 'org', {'version': 1, 'shapes': [], 'tracks': []})
            with self.assertRaises(ValueError):
                annotate_batch.patch_annotations(2, 'org', {'version': 1, 'shapes': [], 'tracks': [], 'action': 'update'})
        evaluator.assert_not_called()

    def test_fetch_json_rejects_non_2xx_responses(self):
        expressions = []
        with patch.object(annotate_batch, 'evaluate', side_effect=lambda expression: expressions.append(expression) or {'ok': True}):
            result = annotate_batch.fetch_json('/api/labels?job_id=2')
        self.assertEqual(result, {'ok': True})
        self.assertIn('r.ok', expressions[0])
        self.assertIn('HTTP ', expressions[0])
        self.assertIn('/api/labels?job_id=2', expressions[0])

    def test_delete_refs_preserves_shapes_and_tracks_outside_processed_span(self):
        inside_shape = existing_shape(100, 10, 5)
        outside_shape = existing_shape(101, 10, 15)
        other_shape = existing_shape(102, 11, 5)
        spanning_track = existing_track(200, 10, 5, [5, 15])
        contained_track = existing_track(201, 10, 5, [6])
        existing = {
            'shapes': [inside_shape, outside_shape, other_shape],
            'tracks': [spanning_track, contained_track],
        }
        shapes, tracks = annotate_batch.delete_refs(existing, {10}, 5, 10)
        self.assertEqual(shapes, [inside_shape])
        self.assertEqual(tracks, [contained_track])
        self.assertIsNot(shapes[0], inside_shape)
        self.assertIsNot(tracks[0], contained_track)
        self.assertIsNot(tracks[0]['shapes'], contained_track['shapes'])

    def test_delete_refs_replaces_selected_tracks_for_full_job(self):
        shape = existing_shape(100, 10, 5)
        track = existing_track(200, 10, 5, [5, 15])
        existing = {
            'shapes': [shape],
            'tracks': [track],
        }
        shapes, tracks = annotate_batch.delete_refs(existing, {10})
        self.assertEqual(shapes, [shape])
        self.assertEqual(tracks, [track])
        self.assertIsNot(shapes[0], shape)
        self.assertIsNot(tracks[0], track)

    def test_apply_keeps_selected_shapes_outside_processed_span(self):
        kept_shape = existing_shape(100, 10, 1)
        replaced_shape = existing_shape(101, 10, 5)
        existing = {
            'version': 1,
            'shapes': [kept_shape, replaced_shape],
            'tracks': [],
            'tags': [],
        }
        calls = []

        def fake_patch(job_id, organization, payload):
            calls.append(payload)
            return {'version': 2}

        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 5, [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json', return_value=existing), \
                    patch.object(annotate_batch, 'patch_annotations', side_effect=fake_patch), \
                    patch.object(annotate_batch, 'evaluate'):
                annotate_batch.apply(2, root, 'org', 5, 5)
        self.assertEqual(calls[-1]['shapes'], [replaced_shape])
        self.assertIsNot(calls[-1]['shapes'][0], replaced_shape)

    def test_apply_url_encodes_organization_in_existing_annotations_fetch(self):
        urls = []

        def fake_fetch(url):
            urls.append(url)
            return {'version': 1, 'shapes': [], 'tracks': [], 'tags': []}

        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json', side_effect=fake_fetch), \
                    patch.object(annotate_batch, 'patch_annotations', return_value={'version': 2}), \
                    patch.object(annotate_batch, 'evaluate'):
                annotate_batch.apply(2, root, 'team one')
        self.assertEqual(urls, ['/api/jobs/2/annotations?org=team+one'])

    def test_zero_predictions_refuse_destructive_writes(self):
        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json') as fetcher, \
                    patch.object(annotate_batch, 'patch_annotations') as patcher:
                with self.assertRaises(ValueError):
                    annotate_batch.apply(2, root, 'org')
        fetcher.assert_not_called()
        patcher.assert_not_called()

    def test_malformed_points_refuse_destructive_writes(self):
        with tempfile.TemporaryDirectory() as root:
            write_predictions(root, 1, [
                {'type': 'rectangle', 'label': 'car', 'points': [0, 0, 10]},
                {'type': 'polygon', 'label': 'road', 'points': [0, 0, 10, 10]},
                {'type': 'rectangle', 'label': 'car', 'points': ['x', 0, 10, 10]},
            ])
            with patch.object(annotate_batch, 'load_label_records', return_value=records()), \
                    patch.object(annotate_batch, 'fetch_json') as fetcher, \
                    patch.object(annotate_batch, 'patch_annotations') as patcher:
                with self.assertRaises(ValueError):
                    annotate_batch.apply(2, root, 'org')
        fetcher.assert_not_called()
        patcher.assert_not_called()

    def test_allowed_labels_file_controls_full_job_selection(self):
        with tempfile.TemporaryDirectory() as root:
            predictions = Path(root) / 'predictions'
            write_predictions(predictions, 1, [{'type': 'rectangle', 'label': 'car', 'points': [0, 0, 4, 4]}])
            (Path(root) / 'allowed_labels.json').write_text(json.dumps(['car', 'person']), encoding='utf-8')
            names = annotate_batch.allowed_labels(predictions)
        self.assertEqual(names, ['car', 'person'])


class PoseSkeletonTests(unittest.TestCase):
    def test_named_sublabels_are_mapped_by_name(self):
        mapping = pose_skeleton.sublabel_mapping(skeleton_records(reverse=True)[0]['sublabels'])
        self.assertEqual(mapping[0], 100)
        self.assertEqual(mapping[16], 116)

    def test_numeric_sublabels_are_mapped_by_number(self):
        mapping = pose_skeleton.sublabel_mapping(skeleton_records(reverse=True, numeric=True)[0]['sublabels'])
        self.assertEqual(mapping[0], 100)
        self.assertEqual(mapping[16], 116)

    def test_incomplete_sublabels_are_rejected(self):
        with self.assertRaises(pose_skeleton.SkeletonSchemaError):
            pose_skeleton.sublabel_mapping([{'id': 1, 'name': 'nose'}])

    def test_malformed_keypoints_are_rejected(self):
        with self.assertRaises(pose_skeleton.SkeletonSchemaError):
            pose_skeleton.skeleton_shape(20, 1, {'keypoints': [[1, 2, 0.9]] * 16}, skeleton_records()[0]['sublabels'])


class UiTests(unittest.TestCase):
    def test_page_only_adds_two_mode_options_and_yolo_request(self):
        style = ui.PAGE.split('<style>', 1)[1].split('</style>', 1)[0]
        self.assertEqual(ui.PAGE.count('<option'), 4)
        self.assertIn('value="tracking"', ui.PAGE)
        self.assertIn('value="skeleton"', ui.PAGE)
        self.assertIn("model:'yolo'", ui.PAGE)
        self.assertNotIn("model:'sam3'", ui.PAGE)
        self.assertIn('.app{display:grid;grid-template-columns:280px 1fr;min-height:100vh}', style)
        self.assertNotIn('tracking', style.lower())
        self.assertNotIn('skeleton', style.lower())

    def test_backend_defaults_and_validation(self):
        self.assertEqual(ui.validate_model(None), 'yolo')
        self.assertEqual(ui.validate_model('yolo'), 'yolo')
        with self.assertRaises(ValueError):
            ui.validate_model('sam3')
        self.assertEqual(ui.validate_mode('tracking'), 'tracking')
        with self.assertRaises(ValueError):
            ui.validate_mode('sam3')

    def test_start_defaults_to_yolo_model(self):
        commands = []

        class FakeProcess:
            def __init__(self, command):
                self.command = command
                self.stdout = []

            def wait(self):
                return 0

        def fake_popen(command, **kwargs):
            commands.append(command)
            return FakeProcess(command)

        with patch.object(ui.subprocess, 'Popen', side_effect=fake_popen):
            ui.start({'url': 'https://cvat.example/tasks/1/jobs/2', 'mode': 'skeleton'}, 'run')
        self.assertIn('yolo', commands[0])
        self.assertNotIn('sam3', commands[0])
        self.assertIn('skeleton', commands[0])


class ConfigTests(unittest.TestCase):
    def test_opencode_config_lists_only_yolo_modes(self):
        path = Path(__file__).resolve().parent.parent / 'opencode.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        agent = config['agent']['yolo-cvat-agent']
        self.assertNotIn('model', config)
        self.assertNotIn('model', agent)
        self.assertEqual(agent['options']['modes'], ['bbox', 'segmentation', 'tracking', 'skeleton'])
        self.assertIn('yolo_agent.infer', agent['options']['flow'])
        self.assertNotIn('sam3', ' '.join(agent['options']['modes'] + agent['options']['flow']).lower())
        self.assertEqual(agent['options']['models']['skeleton'], 'yolo11n-pose.pt')
        self.assertEqual(agent['options']['models']['tracking'], 'yolo11n.pt')


class CdpOriginTests(unittest.TestCase):
    def test_targets_cvat_origin_instead_of_local_agent_tab(self):
        targets = [
            {'type': 'page', 'url': 'http://127.0.0.1:8787/', 'webSocketDebuggerUrl': 'ws://agent'},
            {'type': 'page', 'url': 'http://127.0.0.1:8080/tasks/1/jobs/2', 'webSocketDebuggerUrl': 'ws://cvat'},
        ]
        previous = cdp.ORIGIN
        try:
            cdp.set_origin('http://127.0.0.1:8080/tasks/1/jobs/2')
            with patch.object(cdp, 'cdp_targets', return_value=targets):
                target = cdp.page_target()
        finally:
            cdp.ORIGIN = previous
        self.assertEqual(target['webSocketDebuggerUrl'], 'ws://cvat')

    def test_does_not_fall_back_to_agent_tab_for_missing_local_origin(self):
        targets = [
            {'type': 'page', 'url': 'http://127.0.0.1:8787/', 'webSocketDebuggerUrl': 'ws://agent'},
        ]
        previous = cdp.ORIGIN
        try:
            cdp.set_origin('http://127.0.0.1:8080/tasks/1/jobs/2')
            with patch.object(cdp, 'cdp_targets', return_value=targets):
                with self.assertRaises(RuntimeError):
                    cdp.page_target()
        finally:
            cdp.ORIGIN = previous


if __name__ == '__main__':
    unittest.main()
