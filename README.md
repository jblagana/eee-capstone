# How to Run the Project

## YOLOv8-ByteTrack
```
python -m integration.main
```
> [!TIP]
**arguments**: <br/>
`--yolo-path`       path of YOLO model `default = "./yolo_model/best_finalCustom.pt"`  <br/>
`--persist-yolo`    displays YOLO inference if enabled `default="store_false"` <br/>
`--bytetrack-path`  path of ByteTrack configuration file `default = "./loitering/custom-bytetrack.yaml"` <br/>
`--max-age`         maximum consecutive missed detections before deleting ID `default = 500` <br/>
`--source`          for camera: 0. for video: folder_path where video/s is stored `default="./integration/input-vid"` <br/>
`--device`          device to use `default = 'cuda'`<br/>
`--save-vid`        saves annotated video if enabled `default = 0` <br/>
`--no-display`      disables playing of video while processing <br/>

## YOLOv8-ByteTrack with TensorRT
```
python -m trt_integration.main
```
> [!TIP]
**arguments**: <br/>
`--input`        input type: 'video' or 'webcam' `default = 'video'` <br/>
`--yolomodel`    YOLO model: 'custom', 'v8n', 'v7t', 'v5n' `default = 'custom'` <br/>
`--max-age`      maximum consecutive missed detections before deleting ID `default = 500` <br/>
`--save-vid`     saves annotated video if enabled (1) `default = 0`  <br/>
`--no-display`   disables playing of video while processing <br/>
