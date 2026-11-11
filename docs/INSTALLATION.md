# Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For Docker:

```powershell
docker build -t autolabel-yolov .
```

The first YOLO run downloads `yolo11x.pt` or `yolo11x-seg.pt`. Chrome is opened automatically with the local profile and the user can log in to CVAT.
