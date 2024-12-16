# Pose Detection
Basic pose detection demo using mediapipe and opencv.

# Setup
Make sure you have Python 3.12+ installed.
Install Poetry: `pip install poetry`
Clone this repository and navigate to the project directory.
Run `poetry install` to install the dependencies.

Alternatively, run `pip install -r requirements.txt`

# Usage
```
python3 pose_detection/detect_pose.py --source <path to video file>
```
A live demo with visualization of keypoints and knee angle will be displayed in a window. Press 'q' to exit or stop the script.

Limitations:

The knee angle is calculated based on left hip, left knee and left ankle landmarks. If these points are not detected correctly, the angle calculated will be incorrect. Currently, person detection is not incorporated, hence, only 1 person is expected in the scene.