import streamlit as st
import os
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from agent.tools.recognise import RecogniseTool
from agent.tools.video_cut import VideoCutTool

# Initialize tools and LLM
recognise_tool = RecogniseTool()
video_cut_tool = VideoCutTool()

def initialize_agents():
    supervisor_agent = Agent(
        role='Supervisor',
        goal='Supervise the face recognition and video cutting agents and interact with users',
        backstory="""You are a supervisor who ensures the face recognition and video cutting agents 
        are working correctly. You manage the team and interact with users.
        
        IMPORTANT: You should directly respond to simple greetings like "hi", "hello", etc. without delegating to other agents.
        If users ask about your capabilities, you should explain what you and your team can do without delegating.
        Only delegate to other agents if the user is explicitly requesting face recognition or video processing tasks.
        
        Be polite, friendly, and helpful at all times.""",
        tools=[],
        verbose=True
    )

    face_recognition_agent = Agent(
        role='Face Recognition Specialist',
        goal='Accurately detect and track faces in videos',
        backstory="""You are an expert in computer vision and face recognition technology.""",
        tools=[recognise_tool],
        verbose=True
    )

    video_editor_agent = Agent(
        role='Video Editor',
        goal='Create precise video clips based on timestamp data',
        backstory="""You are a skilled video editor specializing in automated video processing.""",
        tools=[video_cut_tool],
        verbose=True
    )
    
    return supervisor_agent, face_recognition_agent, video_editor_agent

def initialize_tasks(agents, image_path=None, video_path=None, user_message=None):
    supervisor_agent, face_recognition_agent, video_editor_agent = agents
    
    supervisor_task = Task(
        description=f"""
        Process the user message: "{user_message}"
        
        IMPORTANT RULES:
        1. If the user simply greets you (like "hi", "hello", etc.), respond directly with a greeting without delegating.
        2. If the user asks about your capabilities, explain what you and your team can do without delegating.
        3. Only coordinate with other agents if there's an explicit request for face recognition or video processing.
        
        Provide appropriate response to the user.
        """,
        expected_output="Response to user query and coordination results",
        agent=supervisor_agent
    )
    
    if image_path and video_path:
        face_detection_task = Task(
            description=f"""
            1. Process the image at {image_path} and video at {video_path} for face recognition
            2. Return the timestamp data in the required format
            """,
            expected_output="A list of timestamps where the target face appears in the video.",
            agent=face_recognition_agent
        )

        video_cutting_task = Task(
            description=f"""
            1. Process video at {video_path} using the timestamp data
            2. Extract and save video segments
            3. Organize clips by track ID
            """,
            expected_output="Video clips segmented based on timestamps.",
            context=[face_detection_task],
            agent=video_editor_agent
        )
        
        return [supervisor_task, face_detection_task, video_cutting_task]
    else:
        return [supervisor_task]

def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None

def is_simple_interaction(message):
    """Check if the message is a simple greeting or capability question"""
    message = message.lower()
    greeting_keywords = ["hi", "hello", "hey", "greetings"]
    capability_keywords = ["what can you do", "capabilities", "functions", "features", "help"]
    
    is_greeting = any(keyword in message for keyword in greeting_keywords) and len(message.split()) < 3
    is_capability_question = any(keyword in message for keyword in capability_keywords)
    
    return is_greeting or is_capability_question

def main():
    st.set_page_config(page_title="Face Recognition System", layout="wide")
    initialize_session_state()
    
    st.title("🎥 Face Recognition Assistant")
    
    with st.sidebar:
        st.header("Upload Files")
        image_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'])
        video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        
        if image_file:
            st.image(image_file, caption="Reference Image", use_column_width=True)
            st.session_state.image_path = save_uploaded_file(image_file, "temp_uploads")
            
        if video_file:
            st.video(video_file)
            st.session_state.video_path = save_uploaded_file(video_file, "temp_uploads")
        
        st.header("How to Use")
        st.markdown("""
        1. Upload your reference image and video
        2. Chat with the assistant about what you'd like to do
        3. The system can:
           - 👤 Recognize faces
           - ✂️ Cut video segments
           - 💬 Answer questions
        """)

    # Chat interface
    st.subheader("Chat with Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to do?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the message
        with st.chat_message("Assistant"):
            with st.spinner("Processing..."):
                try:
                    # Check if it's a simple interaction (greeting or capability question)
                    if is_simple_interaction(prompt):
                        # Just use the supervisor agent for simple interactions
                        agents = initialize_agents()
                        tasks = initialize_tasks(agents, user_message=prompt)
                        
                        simple_crew = Crew(
                            agents=[agents[0]],  # Only use supervisor
                            tasks=tasks[:1],     # Only use supervisor task
                            process=Process.sequential,
                            verbose=True
                        )
                        
                        response = simple_crew.kickoff(
                            inputs={
                                "user_message": prompt
                            }
                        )
                    else:
                        # Check if files are uploaded for processing tasks
                        if not st.session_state.image_path or not st.session_state.video_path:
                            response = "Please upload both a reference image and a video file before we proceed with processing tasks."
                        else:
                            # Initialize agents and create crew for processing
                            agents = initialize_agents()
                            tasks = initialize_tasks(
                                agents, 
                                st.session_state.image_path, 
                                st.session_state.video_path,
                                prompt
                            )
                            
                            face_tracking_crew = Crew(
                                agents=[agents[0], agents[1], agents[2]],
                                tasks=tasks,
                                manager_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
                                process=Process.hierarchical,
                                memory=True,
                                verbose=True
                            )
                            
                            # Run the processing
                            response = face_tracking_crew.kickoff(
                                inputs={
                                    "user_message": prompt,
                                    "image_path": st.session_state.image_path,
                                    "video_path": st.session_state.video_path,
                                }
                            )
                            
                            # Display any processed clips
                            output_dir = "data/output_clips"
                            if os.path.exists(output_dir):
                                st.subheader("Processed Clips")
                                for clip in os.listdir(output_dir):
                                    if clip.endswith(('.mp4', '.avi', '.mov')):
                                        st.video(os.path.join(output_dir, clip))
                
                except Exception as e:
                    response = f"An error occurred: {str(e)}"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()