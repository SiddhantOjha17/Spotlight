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
    goal='Supervise the face recognition and video cutting agents and interact with users',
    backstory="""You are a supervisor who ensures the face recognition and video cutting agents are working correctly. You have two team mates, 
    one who recognises faces and the other who cuts videos. You are responsible for ensuring that the agents are working correctly and that the output is as expected.
    You are also responsible for providing the agents with the necessary context and instructions.
    You have to talk to the user and answer their questions about the video and the agents.
    
    IMPORTANT: You should directly respond to simple greetings like "hi", "hello", etc. without delegating to other agents.
    If users ask about your capabilities, you should explain what you and your team can do without delegating.
    Only delegate to other agents if the user is explicitly requesting face recognition or video processing tasks.
    
    You have to be polite and friendly.
    """,
    tools=[],
    verbose=True
)

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

supervisor_task = Task(
    description="""
    You have to talk to the user and answer their questions about the video and the agents.
    
    IMPORTANT RULES:
    1. If the user simply greets you (like "hi", "hello", etc.), respond directly with a greeting without delegating.
    2. If the user asks about your capabilities, explain what you and your team can do without delegating.
    3. Only delegate to other agents if there's an explicit request for face recognition or video processing.
    
    You have to be polite, friendly, and helpful.
    """,
    expected_output="Helps the user answer their query, greets them and is polite. Only delegates work when appropriate.",
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

# Main function to process user input
def process_user_input(user_input, input_image, input_video):
    # Check if it's just a greeting or capability question
    greeting_keywords = ["hi", "hello", "hey", "greetings"]
    capability_keywords = ["what can you do", "capabilities", "functions", "features", "help"]
    
    is_greeting = any(keyword in user_input.lower() for keyword in greeting_keywords) and len(user_input.split()) < 3
    is_capability_question = any(keyword in user_input.lower() for keyword in capability_keywords)
    
    if is_greeting or is_capability_question:
        # Use just the supervisor agent for simple interactions
        solo_crew = Crew(
            agents=[supervisor_agent],
            tasks=[supervisor_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = solo_crew.kickoff(inputs={
            "user_message": user_input,
        })
        
        return result
    else:
        # Use the full crew for actual processing tasks
        face_tracking_crew = Crew(
            agents=[supervisor_agent, face_recognition_agent, video_editor_agent],
            tasks=[supervisor_task, face_detection_task, video_cutting_task],
            manager_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            process=Process.hierarchical,
            memory=True,
            verbose=True
        )
        
        result = face_tracking_crew.kickoff(inputs={
            "user_message": user_input,
            "image_path": input_image,
            "video_path": input_video,
        })
        
        return result

# Main execution
if __name__ == "__main__":
    input_image = "D:\PersonalProjects\Spotlight\data\input_images\ishu.jpg"
    input_video = "D:\PersonalProjects\Spotlight\data\input_videos\ishu.mp4"
    
    while True:
        user_input = input("Enter your message: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
            
        result = process_user_input(user_input, input_image, input_video)
        print(result)