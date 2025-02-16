from crewai import Agent, Task, Crew, Process
from tools.recognise import RecogniseTool
from tools.video_cut import VideoCutTool

# Initialize tools
recognise_tool = RecogniseTool()
video_cut_tool = VideoCutTool()

# Create Face Recognition Agent
face_recognition_agent = Agent(
    role='Face Recognition Specialist',
    goal='Accurately detect and track faces in videos',
    backstory="""You are an expert in computer vision and face recognition technology. 
    Your specialty is identifying and tracking specific individuals across video footage.""",
    tools=[recognise_tool],
    verbose=True
)

# Create Video Editor Agent
video_editor_agent = Agent(
    role='Video Editor',
    goal='Create precise video clips based on timestamp data',
    backstory="""You are a skilled video editor specializing in automated video processing.
    Your expertise is in extracting specific segments from videos with high precision.""",
    tools=[video_cut_tool],
    verbose=True
)

# Define Tasks
face_detection_task = Task(
    description="""
    1. Load the input image and video
    2. Detect and track the target face throughout the video
    3. Record all timestamps where the person appears
    4. Return the timestamp data in the required format
    """,
    agent=face_recognition_agent
)

video_cutting_task = Task(
    description="""
    1. Receive timestamp data from face detection
    2. For each appearance:
        - Extract the video segment
        - Save it as a separate clip
    3. Organize clips by track ID
    """,
    agent=video_editor_agent
)

# Create and Run Crew
face_tracking_crew = Crew(
    agents=[face_recognition_agent, video_editor_agent],
    tasks=[face_detection_task, video_cutting_task],
    process=Process.sequential  # Tasks must run in order
)

# Function to run the crew with specific inputs
def process_video(input_image_path, input_video_path, output_directory):
    """
    Process a video to find and clip segments containing a specific person.
    
    Args:
        input_image_path (str): Path to the reference image of the person
        input_video_path (str): Path to the video to analyze
        output_directory (str): Path where output clips should be saved
    """
    
    # Create the context with input paths
    context = {
        "input_image": input_image_path,
        "input_video": input_video_path,
        "output_dir": output_directory
    }
    
    # Run the crew with the context
    result = face_tracking_crew.kickoff(context=context)
    return result

# Example usage
if __name__ == "__main__":
    input_image = "D:\PersonalProjects\Spotlight\data\input_images\ishu.jpg"
    input_video = "D:\PersonalProjects\Spotlight\data\input_videos\ishu.mp4"
    output_dir = "D:\PersonalProjects\Spotlight\data\output_clips"
    
    process_video(input_image, input_video, output_dir)