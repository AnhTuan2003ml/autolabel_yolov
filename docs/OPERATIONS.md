# Operations

Start the web agent:

```powershell
python -m cvat_agent.ui
```

Use **Mở profile đăng nhập** to open the current CVAT job in the controlled Chrome profile. Choose `bbox` or `segmentation`, then run all frames.

The runner reads all paginated labels, downloads all job frames, replaces annotations only for those frames, and uploads the YOLO result. YOLO pretrained models support their trained classes; custom CVAT labels without a matching YOLO class are reported as missing.
