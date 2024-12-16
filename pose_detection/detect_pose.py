import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    Args:
        a: Point 1 (x, y)
        b: Point 2 (x, y) [Vertex point]
        c: Point 3 (x, y)
    Returns:
        Angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors
    ba = a - b
    bc = c - b
    
    # Calculate cosine angle (cos(theta) = (a.b)/(||a||*||b||))
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def detect_pose(video_path):
    """Driver function. Run pose estimation on input video
    
    Args:
        video_path (str): Path to video file or url to input stream
    """
    # Mediapipe initialization
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Video processing
    cap = cv2.VideoCapture(video_path)
    knee_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # Draw pose landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of relevant points
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate knee angle
            angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(angle)

            # Display angle
            cv2.putText(image, f'Knee Angle: {int(angle)}', 
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Output knee angles
    # print(knee_angles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, type=str, help="path to a vdeo file or video stream url.")
    args = parser.parse_args()
    assert args.source is not None, "PLEASE PASS A SOURCE USING --source"
        

    detect_pose(args.source)