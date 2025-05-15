from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize OpenAI's GPT model
llm = ChatOpenAI(model="gpt-4", openai_api_key="Your Open API Key")

# Define a simple prompt template for the coding assistant
code_correction_prompt = """
You are a coding assistant. A user will provide a code snippet, and you will:
1. Detect any syntax or logical errors in the code.
2. Suggest a correction or improvement if necessary.
3. Provide a brief explanation of the error and the correction made.

Code snippet: {code_snippet}
"""

# Create a LangChain prompt template
prompt_template = PromptTemplate(input_variables=["code_snippet"], template=code_correction_prompt)

# Initialize the LLMChain with the prompt template and model
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

def self_correcting_assistant(code_snippet: str) -> str:
    # Run the code through the LLMChain
    corrected_response = llm_chain.invoke(code_snippet=code_snippet)

    return corrected_response

# Test with an incorrect Python code snippet
code = """
def add_numbers(a, b):
    return a + b
print(add_numbers(1, 2))  # Intentional typo in function call
"""

# Get the corrected response
response = self_correcting_assistant(code)
print(response)

from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType

# Define tools, such as debugging tools or additional code-checking utilities
tools = [
    Tool(
        name="CodeCorrectionTool",
        func=self_correcting_assistant,
        description="A tool to suggest corrections for code snippets"
    )
]

# Initialize the agent with the tools
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Run the agent
agent_response = agent.run("Fix the following Python code with a syntax error: print('Hello World'")  # Intentional typo
print(agent_response)
