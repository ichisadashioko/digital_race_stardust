- `unity_recorder.py` uses [`20190111_sobel_lane_detect.py`](../20190111_sobel_lane_detect.py)'s pipeline to control car.
- `Video_Writer` class in [`video_recorder.py`](video_recorder.py) creates a new `VideoWriter` with name is the timestamp (format `yyyyddmm_hhmmss.avi`) that the object is initialized if `name` is not passed upon initialization and video frame's dimension is 320x240 (width x height).
- The recorder video will be saved at `/home/<username>/Videos` folder
    - Usage:
    ```python
    video_writer = Video_Writer()
    # or video_writer = Video_Writer(name='video_name') for custom name
    # or video_writer = Video_Writer(dimension=(480,320)) for custom dimension 480px width and 320px height

    # add image frame to the video
    video_writer.write(frame)

    # release (save) the video_writer (properly)
    video_writer.release()
    ```