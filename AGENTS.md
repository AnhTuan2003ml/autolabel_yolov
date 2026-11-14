# Shared Repository Context

This file is the canonical shared context for `cxa`, `cxb`, and OpenCode when they work in this repository. Read it before changing code. Update only the durable handoff state below when architecture, commands, constraints, or verified repository state changes. Do not store chat transcripts, credentials, job data, browser profiles, or model weights here.

## Scope and Architecture

This is the standalone YOLO CVAT agent. It is independent from the parent SAM3 application. `cvat_agent/` owns CVAT browser access, job metadata, frame download, orchestration, annotation conversion, and upload. `yolo_agent/infer.py` is the only inference adapter and supports `bbox`, `segmentation`, `tracking`, and `skeleton`. The UI must route all four modes with `model: yolo` and must not start or call a SAM3 worker.

Predictions use JSON records with `type`, `label`, `confidence`, and `points`. Tracking records also use `track_id`. Skeleton records use 17 COCO keypoints. Preserve annotations for labels outside the selected run. Refuse annotation mutation when output is empty or invalid.

## Working Rules

- Always use `graphify` for code relationship inspection.
- Keep diffs limited to code, configuration, and tests required by the request.
- Do not add, edit, remove, or move comments, docstrings, TODOs, temporary annotations, or explanatory text inside source code.
- Preserve all existing comments and docstrings.
- Preserve unrelated worktree changes and do not access live CVAT data, credentials, browser profiles, or model weights during tests.
- Keep model adapters independent from CVAT transport and annotation writing.

## Commands

- Start UI: `python -m cvat_agent.ui`
- Run job: `python -m cvat_agent.job_runner <job-url> <organization> yolo <bbox|segmentation|tracking|skeleton> runs 60`
- Compile: `python -m compileall -q cvat_agent yolo_agent`
- Test: `python -m unittest discover -s tests`
- Inspect relationships: `graphify query "trace CVAT mode to YOLO adapter and annotation upload" --graph graphify-out/graph.json`

## Durable Handoff

- The supported model is YOLO only.
- The supported modes are bbox, segmentation, tracking, and skeleton.
- Tracking uses YOLO track IDs and CVAT tracks.
- Skeleton uses YOLO pose output and requires a compatible 17-keypoint CVAT skeleton label.
- The four-mode YOLO CVAT integration is implemented and verified by `python -m compileall -q cvat_agent yolo_agent` and `python -m unittest discover -s tests`.
