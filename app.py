from crewai import Agent, Task, Crew, Process
from crewai_tools import YoutubeChannelSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Initialize YouTube Tool
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle=' ')

# Step 1: Creating the Blog Researcher Agent
blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video transcription for the topic {topic} from the provided YT channel',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning, and Gen AI, "
        "and providing suggestions."
    ),
    tools=[yt_tool],
    allow_delegation=True
)

# Step 2: Creating the Blog Writer Agent
blog_writer = Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft "
        "engaging narratives that captivate and educate, bringing new "
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False
)

# Step 3: Defining Tasks

# Research Task
research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel."
    ),
    expected_output='A comprehensive 3-paragraph report based on the {topic} of video content.',
    tools=[yt_tool],
    agent=blog_researcher,
)

# Writing Task
write_task = Task(
    description=(
        "Get the info from the YouTube channel on the topic {topic}."
    ),
    expected_output='Summarize the info from the YouTube video on the topic {topic} and create blog content.',
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md'
)

# Step 4: Forming the Crew and Process Configuration
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Step 5: Executing the Process with Enhanced Feedback
result = crew.kickoff(inputs={'topic': 'AI vs ML vs DL vs Data Science'})
print(result)
