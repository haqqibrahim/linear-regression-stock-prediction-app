import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def model_analysis_agent(model_insight):
    model_analysis_agent = Agent(
        name="Model Analysis",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY),
        instructions=["You are a quant analyst, you are given a model insight and you need to analyze it."],
        show_tool_calls=True,
        markdown=True,
    )
    response = model_analysis_agent.run(model_insight, stream=False)
    return response.content


