import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize counters for correct and incorrect squats
correct_squats = 0
incorrect_squats = 0
state = "up"  # Start in the "up" position (standing)
frame_count = 0

# Define thresholds
min_flexion_threshold = 130   # Adjusted threshold to ignore smaller flexion
correct_squat_depth = 90      # Proper squat depth
stand_threshold = 160         # Angle for standing position
buffer_frames = 15            # Increased buffer for preventing multiple counts

# Define function to calculate the angle between three points (landmarks)
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

angles = []
last_feedback = ""  # Store the last feedback
last_squat_result = ""  # Store the last squat result (correct or incorrect)
state = "standing"  # Initial state is standing
buffer_count = 0  # Buffer to prevent multiple detections
buffer_frames = 15  # Buffer to avoid multiple detections at the bottom

# Thresholds for squat analysis
min_flexion_threshold = 120  # Angle for moving down from standing
correct_squat_depth = 90     # Ideal angle for a correct squat
stand_threshold = 160        # Angle for standing position

def squat_detector(frame):
    global correct_squats, incorrect_squats, state, buffer_count, last_feedback, last_squat_result

    # Convert the image to RGB as required by MediaPipe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    # Draw landmarks on the frame if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Get relevant landmarks for squat detection
        landmarks = results.pose_landmarks.landmark
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        # Calculate the angle between hip, knee, and ankle
        squat_angle = calculate_angle(hip, knee, ankle)

        # Squat detection logic
        if state == "standing":
            if squat_angle < min_flexion_threshold:  # The user is starting to squat
                state = "squatting"
                buffer_count = 0  # Reset the buffer
                last_feedback = "Starting to squat."

        elif state == "squatting":
            if squat_angle <= correct_squat_depth:  # Deep enough to count as a correct squat
                state = "bottom"
                last_feedback = "Good depth achieved!"
            elif squat_angle > stand_threshold and buffer_count > buffer_frames:  # The user stood back up
                incorrect_squats += 1
                last_feedback = "Squat was too shallow. Bend deeper!"
                state = "standing"

        elif state == "bottom":
            if squat_angle > stand_threshold and buffer_count > buffer_frames:  # Coming up from the bottom of a squat
                correct_squats += 1
                last_feedback = "Correct squat counted!"
                state = "standing"

        # Display the squat angle near the knee
        cv2.putText(frame, f'Angle: {int(squat_angle)}', 
                    (int(knee.x * frame.shape[1]), int(knee.y * frame.shape[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Update buffer count for each frame in the squatting state
        buffer_count += 1

    # Display squat counts on the video frame
    cv2.putText(frame, f'Correct Squats: {correct_squats}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Incorrect Squats: {incorrect_squats}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the last feedback on the video frame
    if last_feedback:
        cv2.putText(frame, last_feedback, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


# Route to handle video feed from live camera or uploaded video
def gen(camera_source):
    cap = cv2.VideoCapture(camera_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for squat detection and visualization
        processed_frame = squat_detector(frame)
        
         # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Return the frame in a format compatible with Flask's response streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to handle video feed based on live camera or uploaded video
@app.route('/video_feed/<source>')
def video_feed(source):
    if source == "camera":
        return Response(gen(0), mimetype='multipart/x-mixed-replace; boundary=frame')  # Live camera feed
    elif source == "upload":
        return Response(gen(f'{app.config["UPLOAD_FOLDER"]}/uploaded_video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')  # Uploaded video feed

# Route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global correct_squats, incorrect_squats, state, last_feedback, buffer_count
    
    # Reset counters and states when a new video is uploaded
    correct_squats = 0
    incorrect_squats = 0
    state = "standing"
    last_feedback = ""
    buffer_count = 0

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4'))
        return redirect(url_for('video_feed', source='upload'))


# Main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
