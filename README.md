# autolabel_yolov

Standalone YOLO11 agent for object detection and instance segmentation. This project is independent from the SAM3 agent in the parent repository.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Detection

```powershell
python -m yolo_agent.infer --source images --output runs\bbox --mode bbox --model yolo11x.pt
```

## Segmentation

```powershell
python -m yolo_agent.infer --source images --output runs\seg --mode segmentation --model yolo11x-seg.pt
```

Output JSON contains `label`, `confidence`, `type`, and `points`. YOLO downloads the selected model on first use.

## Docker

```powershell
docker build -t autolabel-yolov .
docker run --rm --gpus all -v ${PWD}:/work autolabel-yolov python -m yolo_agent.infer --source /work/images --output /work/runs/seg --mode segmentation --model yolo11x-seg.pt
```

## Verify

```powershell
python -m compileall -q yolo_agent
```

## CVAT full flow

The CVAT runner keeps the same browser/CDP flow as the parent agent. It reads the job labels, downloads every frame, maps YOLO classes to available CVAT labels, removes annotations for processed frames, opens locked or hidden labels, and uploads bbox or segmentation results.

```powershell
python -m cvat_agent.ui
```

Or run directly:

```powershell
python -m cvat_agent.job_runner "https://cvat.edufun.ai/tasks/56/jobs/130" cohort-5a yolo bbox runs 80
```
