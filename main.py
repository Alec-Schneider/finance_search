import streamlit as st
import os
from langchain import PromptTemplate, OpenAI
import yaml
from llm import FinanceLLM

# with open(r"..\openai.yaml") as f:
#     spec = yaml.safe_load(f)
#     key = spec['openai']['key']
#     serp_key = spec['serpapi']['key']

# os.environ['OPENAI_API_KEY'] = key    
# os.environ['SERPAPI_API_KEY'] = serp_key




st.set_page_config(page_title="Finance Search", page_icon=":chart_with_upwards_trend:")
st.header("Finance Search")

st.write("You can search for information about stocks, bonds, ETFs, mutual funds, cryptos, properties, and derivatives")
st.write("You can also search for financial and economic data")

st.markdown("## How to use this app")
st.write("1. Type in your question in the text box below. For example: What is the price of AAPL on 2023-07-07?")
st.write("2. Press the search button to submit your question")

# with col2:  
#     st.image(image="./nyse.png", caption="NYSE", width=200)  


def get_text():
    input_text = st.text_area(label="", placeholder="Enter your question here", key="q_input", height=100)
    return input_text

st.markdown("## Ask a question")

input_text = get_text()
if input_text:
    st.write("You entered: ")
    st.write(input_text)

    llm = FinanceLLM()
    llm.load_agent()
    llm.load_prompt()
    print(llm.agent)
    result = llm.agent.run(llm.prompt.format(question=input_text))

    st.write("Here is the answer:")
    st.write(result)