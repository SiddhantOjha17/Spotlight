from crewai import Agent, Task, Crew, Process, LLM
from tools.recognise import RecogniseTool
from tools.video_cut import VideoCutTool
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

# Initialize tools
recognise_tool = RecogniseTool()
video_cut_tool = VideoCutTool()

llm = LLM(model="gpt-4o-mini")

# Create a Supervisor Agent
supervisor_agent = Agent(
    role='Supervisor',
    goal='Supervise the face recognition and video cutting agents',
    backstory="""You are a supervisor who ensures the face recognition and video cutting agents are working correctly. You have two team mates, 
    one who recognises faces and the other who cuts videos. You are responsible for ensuring that the agents are working correctly and that the output is as expected.
    You are also responsible for providing the agents with the necessary context and instructions.
    You have to talk to the user and answer their questions about the video and the agents.
    If the user greets you, you should greet them back.
    If the user asks you about the agents, you should tell them about their roles and responsibilities.
    You have to be polite and friendly.
    """,
    tools=[],
    # llm=llm,
    verbose=True
)

# Create Face Recognition Agent
face_recognition_agent = Agent(
    role='Face Recognition Specialist',
    goal='Accurately detect and track faces in videos',
    backstory="""You are an expert in computer vision and face recognition technology. 
    Your specialty is identifying and tracking specific individuals across video footage.""",
    tools=[recognise_tool],
    # llm=llm,
    verbose=True
)

# Create Video Editor Agent
video_editor_agent = Agent(
    role='Video Editor',
    goal='Create precise video clips based on timestamp data',
    backstory="""You are a skilled video editor specializing in automated video processing.
    Your expertise is in extracting specific segments from videos with high precision.""",
    tools=[video_cut_tool],
    # llm=llm,
    verbose=True
)

# Define Tasks

supervisor_task = Task(
    description="""
    You have to talk to the user and answer their questions about the video and the agents.
    If the user greets you, you should greet them back.
    If the user asks you about the agents, you should tell them about their roles and responsibilities.
    You have to be polite and friendly.
    You have to be helpful and provide the user with the information they need.
    If the user asks you to do a task, you have to delegate it to the appropriate agent.
    """,
    expected_output="Helps the user answer their query, greets them and is polite. Delegates the work and chooses approprite agent to call",
    agent=supervisor_agent
)

face_detection_task = Task(
    description="""
    1. Pass the inputs, image path at {image_path} and video at {video_path} for the face recognition task.
    2. Wait for the tool to finish it work
    3. Return the timestamp data in the required format
    """,
    expected_output="A list of timestamps where the target face appears in the video.",
    agent=face_recognition_agent
)

video_cutting_task = Task(
    description="""
    1. Receive timestamp data from face detection
    2. Input path for the video is at {video_path}
    3. For each appearance:
        - Extract the video segment
        - Save it as a separate clip
    3. Organize clips by track ID
    """,
    expected_output="Video clips segmented based on timestamps, organized by track ID.",
    context=[face_detection_task],
    agent=video_editor_agent
)

# Create and Run Crew
face_tracking_crew = Crew(
    agents=[face_recognition_agent, video_editor_agent],
    tasks=[face_detection_task, video_cutting_task],
    # manager_agent=supervisor_agent,
    manager_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    process=Process.hierarchical,
    memory=True,
    verbose=True
)
input_image = "D:\PersonalProjects\Spotlight\data\input_images\ishu.jpg"
input_video = "D:\PersonalProjects\Spotlight\data\input_videos\ishu.mp4"

# while True:
user_input = input("Enter your message: ").lower()

inputs = {
            "user_message": f"{user_input}",
            "image_path":input_image,
            "video_path":input_video,
        }

print(inputs)

result = face_tracking_crew.kickoff(inputs=inputs)
print(result)
# =
# if __name__ == "__main__":
#     input_image = "D:\PersonalProjects\Spotlight\data\input_images\ishu.jpg"
#     input_video = "D:\PersonalProjects\Spotlight\data\input_videos\ishu.mp4"
#     output_dir = "D:\PersonalProjects\Spotlight\data\output_clips"
    
#     process_video(input_image, input_video)