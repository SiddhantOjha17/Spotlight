import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# Initialize Face Analysis
app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=3)

# Dictionary to store appearance data for each track
appearances = {}
# Dictionary to store current appearance start times
current_appearances = {}

def format_timedelta(td):
    """Convert timedelta to HH:MM:SS format"""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Load and Process Input Image
input_img = cv2.imread('D:\PersonalProjects\Spotlight\data\input_images\ishu.jpg')
input_faces = app.get(input_img)
if not input_faces:
    print("No face detected in input image.")
    exit()
input_embedding = input_faces[0].embedding / np.linalg.norm(input_faces[0].embedding)

# Process Video Frame-by-Frame
cap = cv2.VideoCapture('D:\PersonalProjects\Spotlight\data\input_videos\ishu.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = timedelta(seconds=frame_count/fps)
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Resize frame for better detection
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height))

    faces = app.get(frame)
    
    detections = []
    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        similarity = cosine_similarity([input_embedding], [emb])[0][0]
        
        if similarity > 0.5:
            bbox = face.bbox.astype(int)
            confidence = similarity
            detections.append(([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], confidence, emb))

    tracks = tracker.update_tracks(detections, frame=frame)

    # Update appearances
    active_tracks = set()
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        active_tracks.add(track_id)
        
        # If this is a new appearance for this track
        if track_id not in current_appearances:
            current_appearances[track_id] = current_time
            if track_id not in appearances:
                appearances[track_id] = []
        
        ltrb = track.to_ltrb()
        x1, y1 = max(0, int(ltrb[0])), max(0, int(ltrb[1]))
        x2, y2 = min(frame.shape[1], int(ltrb[2])), min(frame.shape[0], int(ltrb[3]))
        
        # Draw visualization
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
        text = f"ID {track_id} - {format_timedelta(current_time)}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Check for tracks that have disappeared
    for track_id in list(current_appearances.keys()):
        if track_id not in active_tracks:
            start_time = current_appearances[track_id]
            duration = current_time - start_time
            appearances[track_id].append({
                'start_time': format_timedelta(start_time),
                'end_time': format_timedelta(current_time),
                'duration': format_timedelta(duration)
            })
            del current_appearances[track_id]

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Handle any remaining active appearances at the end of the video
for track_id, start_time in current_appearances.items():
    duration = current_time - start_time
    appearances[track_id].append({
        'start_time': format_timedelta(start_time),
        'end_time': format_timedelta(current_time),
        'duration': format_timedelta(duration)
    })

# Print appearance data
print("\nAppearance Times:")
for track_id, track_appearances in appearances.items():
    print(f"\nTrack ID: {track_id}")
    for idx, appearance in enumerate(track_appearances, 1):
        print(f"Appearance {idx}:")
        print(f"  Start Time: {appearance['start_time']}")
        print(f"  End Time: {appearance['end_time']}")
        print(f"  Duration: {appearance['duration']}")

cap.release()
cv2.destroyAllWindows()