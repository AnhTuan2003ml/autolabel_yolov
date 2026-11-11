FROM ultralytics/ultralytics:latest

WORKDIR /work
COPY yolo_agent /work/yolo_agent
COPY cvat_agent /work/cvat_agent
COPY pyproject.toml requirements.txt /work/

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/work
