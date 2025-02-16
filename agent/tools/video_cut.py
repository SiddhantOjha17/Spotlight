import cv2
import os
from datetime import datetime, timedelta

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class VideoCutToolInput(BaseModel):
    input_video_path: str = Field(description="The path to the video file to be cut")
    output_path: str = Field(description="The path to the output directory")
    start_time: str = Field(description="The start time of the clip in HH:MM:SS format")
    end_time: str = Field(description="The end time of the clip in HH:MM:SS format")
    track_id: int = Field(description="The ID of the tracked person")
    appearance_num: int = Field(description="The appearance number for this track")

class VideoCutTool(BaseTool):
    name: str = "VideoCutTool"
    description: str = "A tool to cut a video based on start and end times"
    args_schema: Type[BaseModel] = VideoCutToolInput

    def _run(self, input_video_path: str, output_path: str, start_time: str, end_time: str, track_id: int, appearance_num: int):
        """
        Extract a clip from the video based on start and end times
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(input_video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Convert times to frame numbers
        start_frame = int(self.time_to_seconds(start_time) * fps)
        end_frame = int(self.time_to_seconds(end_time) * fps)
        
        # Create output filename
        output_filename = f"person_{track_id}_appearance_{appearance_num}.mp4"
        output_file = os.path.join(output_path, output_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Set frame position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Write the frame
            out.write(frame)
            current_frame += 1
            
            # Optional: Show progress
            if current_frame % 30 == 0:  # Update every 30 frames
                progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                print(f"\rProcessing clip {output_filename}: {progress:.1f}%", end="")
        
        # Release resources
        cap.release()
        out.release()
        print(f"\nSaved clip: {output_filename}")
        return output_file

    @staticmethod
    def time_to_seconds(time_str):
        """Convert HH:MM:SS string to seconds"""
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

def create_clip(input_video_path, output_path, start_time, end_time, track_id, appearance_num):
    """
    Extract a clip from the video based on start and end times
    
    Parameters:
    - input_video_path: path to the source video
    - output_path: directory where clips will be saved
    - start_time: clip start time in HH:MM:SS format
    - end_time: clip end time in HH:MM:SS format
    - track_id: ID of the tracked person
    - appearance_num: appearance number for this track
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Convert times to frame numbers
    start_frame = int(VideoCutTool.time_to_seconds(start_time) * fps)
    end_frame = int(VideoCutTool.time_to_seconds(end_time) * fps)
    
    # Create output filename
    output_filename = f"person_{track_id}_appearance_{appearance_num}.mp4"
    output_file = os.path.join(output_path, output_filename)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Set frame position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write the frame
        out.write(frame)
        current_frame += 1
        
        # Optional: Show progress
        if current_frame % 30 == 0:  # Update every 30 frames
            progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
            print(f"\rProcessing clip {output_filename}: {progress:.1f}%", end="")
    
    # Release resources
    cap.release()
    out.release()
    print(f"\nSaved clip: {output_filename}")

def process_appearances(appearances, input_video_path, output_path):
    """
    Process all appearances and create respective video clips
    
    Parameters:
    - appearances: dictionary of appearances by track_id
    - input_video_path: path to the source video
    - output_path: directory where clips will be saved
    """
    for track_id, track_appearances in appearances.items():
        for idx, appearance in enumerate(track_appearances, 1):
            start_time = appearance['start_time']
            end_time = appearance['end_time']
            
            print(f"\nProcessing Track ID {track_id}, Appearance {idx}")
            print(f"Time range: {start_time} to {end_time}")
            
            create_clip(
                input_video_path=input_video_path,
                output_path=output_path,
                start_time=start_time,
                end_time=end_time,
                track_id=track_id,
                appearance_num=idx
            )

# # Example usage
# if __name__ == "__main__":
#     # Sample appearances dictionary (you'll get this from the previous tracking code)
#     appearances = {
#         1: [
#             {
#                 'start_time': '00:00:10',
#                 'end_time': '00:00:15',
#                 'duration': '00:00:05'
#             }
#         ]
#     }
    
#     input_video = 'D:\PersonalProjects\Spotlight\data\input_videos\ishu.mp4'
#     output_dir = 'D:\PersonalProjects\Spotlight\data\output_clips'
    
#     process_appearances(appearances, input_video, output_dir)