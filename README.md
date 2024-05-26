# How to Run the Project

## YOLOv8-ByteTrack
```
python -m integration.main
```
> [!TIP]
**arguments**: <br/>
`--yolo-path`       path of YOLO model `default = "./yolo_model/best_finalCustom.pt"`  <br/>
`--bytetrack-path`  path of ByteTrack configuration file `default = "./loitering/custom-bytetrack.yaml"` <br/>
`--max-age`         maximum consecutive missed detections before deleting ID `default = 500` <br/>
`--source`          for camera: 0. for video: folder_path where video/s is stored `default="./integration/input-vid"` <br/>
`--device`          device to use `default = 'cuda'`<br/>
`--save-vid`        saves annotated video if enabled `default = 0` <br/>
`--no-display`      disables playing of video while processing <br/>

```
python -m integration.main_skipping --skip-frames 4
```

## YOLOv8-ByteTrack with TensorRT
```
LD_PRELOAD=/home/robbers/venv/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0    # preload libgomp
python -m trt_integration.main
```
> [!TIP]
**arguments**: <br/>
`--input`        input type: 'video' or 0/1 (CSI camera) `default = 'video'` <br/>
`--max-age`      maximum consecutive missed detections before deleting ID `default = 500` <br/>
`--skip-frames`  enables skipping of frames by input number `default = 1` (no skip) <br/>
`--no-display`   disables playing of video while processing <br/>
`--no-profile`   disables profiling of code <br/>
`--no-fps-log`   disables logging of fps <br/>
`--no-annotate`  disables annotation of frame <br/>

## Displaying Evaluation Plots (FPS, resource consumption, profiles)
```
python -m integration.profiling.display
python -m trt_integration.profiling.display
```
