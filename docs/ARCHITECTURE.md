# Architecture

The YOLO project preserves the CVAT flow of the parent agent while replacing only the model adapter.

```text
ui.py
  -> job_runner.py
      -> cdp.py and job_data.py
      -> label_agent.py
      -> yolo_agent/infer.py
      -> annotate_batch.py
```

`job_runner.py` controls the task lifecycle. `cdp.py` manages the browser profile and CVAT API session. `job_data.py` downloads frames. `yolo_agent/infer.py` produces bbox or polygon JSON. `annotate_batch.py` removes shapes for processed frames, opens hidden or locked labels, and uploads the replacement annotation.
