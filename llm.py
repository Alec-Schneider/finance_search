from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools


template = """
    Below is a question in regards to a stock or financial markets. Please answer the question to the best of your ability.
    If the question is not about a stock or any traded security such as bond, etf, mutual fund, crypto,
    property, derivatives, etc, please inform the person asking the question that this is not the right.

    Here are some examples of questions and answers?
    - What is the price of AAPL on 2023-07-07? $190.98
    - What is the yield of the 10 year United States treasury bond on 2023-07-07? 4.06%
    - What is the price of a Cal Ripken Jr. rookie card? Cal Ripken Jr. rookie card is not a traded security.
    - What is the weather in Dallas today? Weather is not searchable here, please ask about financial markerts
    - What is Joe Biden's strategy in Ukraine? Poliitical questions are not searchable here, please ask about financial markets

    Below is the question:
    QUESTION: {question}

    YOUR RESPONSE:
    """



class FinanceLLM:
    def __init__(self):
        
        self.template = template
        self.prompt = None
        self.llm = None
    
    def load_llm(self, temperature=0.0, **kwargs):
        self.llm = OpenAI(temperature=temperature, **kwargs)
        return self.llm
    
    def load_agent(self, tools=["serpapi", "llm-math"]):
        if self.llm is None:
            self.load_llm()
        self.tools = load_tools(tools, llm=self.llm)
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        return self.agent

    def load_prompt(self):
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["question"],
        )
        return self.prompt