import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class RecogniseToolInput(BaseModel):
    image_path: str = Field(description="The path to the image file to be recognised")
    video_path: str = Field(description="The path to the video file to be recognised")

class RecogniseTool(BaseTool):
    name: str = "RecogniseTool"
    description: str = "A tool to recognise faces in an image or video"
    args_schema: Type[BaseModel] = RecogniseToolInput

    def __init__(self):
        super().__init__()
        # Initialize Face Analysis
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # Initialize DeepSORT Tracker
        self.tracker = DeepSort(max_age=30, n_init=3)
        # Dictionary to store appearance data for each track
        self.appearances = {}
        # Dictionary to store current appearance start times
        self.current_appearances = {}

    def format_timedelta(self, td):
        """Convert timedelta to HH:MM:SS format"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _run(self, image_path: str, video_path: str):
        # Load and Process Input Image
        input_img = cv2.imread(image_path)
        input_faces = self.app.get(input_img)
        if not input_faces:
            return "No face detected in input image."
        
        input_embedding = input_faces[0].embedding / np.linalg.norm(input_faces[0].embedding)

        # Process Video Frame-by-Frame
        cap = cv2.VideoCapture(video_path)
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

            faces = self.app.get(frame)
            
            detections = []
            for face in faces:
                emb = face.embedding / np.linalg.norm(face.embedding)
                similarity = cosine_similarity([input_embedding], [emb])[0][0]
                
                if similarity > 0.5:
                    bbox = face.bbox.astype(int)
                    confidence = similarity
                    detections.append(([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], confidence, emb))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Update appearances
            active_tracks = set()
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                active_tracks.add(track_id)
                
                # If this is a new appearance for this track
                if track_id not in self.current_appearances:
                    self.current_appearances[track_id] = current_time
                    if track_id not in self.appearances:
                        self.appearances[track_id] = []

            # Check for tracks that have disappeared
            for track_id in list(self.current_appearances.keys()):
                if track_id not in active_tracks:
                    start_time = self.current_appearances[track_id]
                    duration = current_time - start_time
                    self.appearances[track_id].append({
                        'start_time': self.format_timedelta(start_time),
                        'end_time': self.format_timedelta(current_time),
                        'duration': self.format_timedelta(duration)
                    })
                    del self.current_appearances[track_id]

        # Handle any remaining active appearances at the end of the video
        for track_id, start_time in self.current_appearances.items():
            duration = current_time - start_time
            self.appearances[track_id].append({
                'start_time': self.format_timedelta(start_time),
                'end_time': self.format_timedelta(current_time),
                'duration': self.format_timedelta(duration)
            })

        cap.release()

        # Format the results
        result = "Appearance Times:\n"
        for track_id, track_appearances in self.appearances.items():
            result += f"\nTrack ID: {track_id}\n"
            for idx, appearance in enumerate(track_appearances, 1):
                result += f"Appearance {idx}:\n"
                result += f"  Start Time: {appearance['start_time']}\n"
                result += f"  End Time: {appearance['end_time']}\n"
                result += f"  Duration: {appearance['duration']}\n"

        return result