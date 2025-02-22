import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from agent.tools.recognise import RecogniseTool
from agent.tools.video_cut import VideoCutTool
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize tools
recognise_tool = RecogniseTool()
video_cut_tool = VideoCutTool()

llm = LLM(model="gpt4o", api_key=OPENAI_API_KEY)

# Create a Supervisor Agent
supervisor_agent = Agent(
    role='Supervisor',
    goal='Supervise the face recognition and video cutting agents',
    backstory="""You ensure the face recognition and video cutting agents work correctly.
    You assign tasks, provide context, and answer user queries politely and helpfully.""",
    tools=[],
    # llm=llm,
    verbose=True
)

# Create Face Recognition Agent
face_recognition_agent = Agent(
    role='Face Recognition Specialist',
    goal='Accurately detect and track faces in videos',
    backstory="""Expert in computer vision and face recognition technology, identifying and tracking faces across video footage.""",
    tools=[recognise_tool],
    # llm=llm,
    verbose=True
)

# Create Video Editor Agent
video_editor_agent = Agent(
    role='Video Editor',
    goal='Create precise video clips based on timestamp data',
    backstory="""Skilled video editor specializing in automated video processing and extracting precise video segments.""",
    tools=[video_cut_tool],
    # llm=llm,
    verbose=True
)

# Define Tasks
supervisor_task = Task(
    description="""Answer user queries about the agents and delegate tasks appropriately.""",
    expected_output="Provides user information and delegates work to agents.",
    agent=supervisor_agent
)

face_detection_task = Task(
    description="""Detect and track faces in video, return timestamps where the target face appears.""",
    expected_output="List of timestamps where the target face appears.",
    agent=face_recognition_agent
)

video_cutting_task = Task(
    description="""Extract video clips based on detected timestamps and organize them by track ID.""",
    expected_output="Video clips segmented and organized by track ID.",
    agent=video_editor_agent
)

# Streamlit UI
st.title("Agentic Video Face Tracking")
st.write("Upload a video and an image of the person to track in the video.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
uploaded_image = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

if st.button("Process Video") and uploaded_video and uploaded_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
        video_file.write(uploaded_video.read())
        video_path = video_file.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as image_file:
        image_file.write(uploaded_image.read())
        image_path = image_file.name
    
    # Create and Run Crew
    face_tracking_crew = Crew(
        agents=[supervisor_agent, face_recognition_agent, video_editor_agent],
        tasks=[supervisor_task, face_detection_task, video_cutting_task],
        manager_agent=supervisor_agent,
        process=Process.hierarchical,
        verbose=True
    )

    result = face_tracking_crew.kickoff()
    st.success("Processing complete!")
    st.write("Results:", result)
    
    # Cleanup temp files
    os.remove(video_path)
    os.remove(image_path)
