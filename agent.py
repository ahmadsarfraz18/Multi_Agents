from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
import os

load_dotenv()
set_tracing_disabled(True)

provider = AsyncOpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash-exp",
    openai_client= provider,
)

# Create the agent

web_dev = Agent(
    name = "Web Developement Expert",
    instructions= "Create responsive and performant websites using modern frameworks and libraries.",
    model = model,
    handoff_description= "Handoff to web developer if the task is related to web development"
)

mobile_app_dev = Agent(
    name = "Mobile App development Expert",
    instructions= "Build Mobile Apps using modern frameworks and libraries.",
    model = model,
    handoff_description= "Handoff to web app developer if the task is related to web app development"
)

digital_marketing = Agent(
    name = "Digital Marketing Expert",
    instructions= "Create engaging and effective digital marketing strategies.",
    model = model,
    handoff_description= "Handoff to digital marketing expert if the task is related to digital marketing."
)

async def myAgent(user_input):
    manager = Agent(
        name = "Manager",
        instructions= "Decide which agent should be used to handle the task.",
        model = model,
        handoffs= [web_dev, mobile_app_dev, digital_marketing]
    ) 

    response = await Runner.run(
        manager,
        input = user_input
    )

    return response.final_output


