import streamlit as st
import openai
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper, wikipedia
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler

# Set up Streamlit app 
st.set_page_config(page_title="Text to Math Promblem Solver", page_icon="$")
st.title("Text to Math Promblem solver Using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="Groq api key", type="password")

if not groq_api_key:
  st.info("Please add your Groq API Key to continue..")
  st.stop()

llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=groq_api_key)

# Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
  name = "Wikipedia",
  func = wikipedia_wrapper.run,
  description = " A tool for searching the Internet to find the various information on the topics mentioned"
)

# Initializing the Math Tool

math_chain = LLMMathChain.from_llm(llm)
calculator = Tool(
  name = "Calculator",
  func = math_chain.run,
  description = "A tool for answering math related questions. Only input mathematical wxpression need to be provided."
)

# Prompt
prompt_template = """
Your a agent tasked for solving users mathematical questions. Logically arrive at the solution and display it point wise for the question below:
Question :{question}
"""

Prompt = PromptTemplate(
  input_variables = ["question"],template = prompt_template
)

# Combine all the tools & chain
chain = LLMChain(llm, Prompt)

reasoning_tool = Tool(
  name = "Reasoning",
  func = chain.run,
  description = "A tool for nswering logic-based and reasoning questions."
)

# Intializing the Agents
assistant_agent = initialize_agent(
  tools = [wikipedia_tool,calculator,reasoning_tool],
  llm = llm,
  agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose = False,
  handle_parsing_error = True
)

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    {"role" : "assistant", "content" : "Hi, I am A Math chatbot who can answer all yours math question"}

  ]

for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

# Lets start the interaction
question = st.text_area("Enter your question :", "I have 5 bananas and 7 grapes total how many fruits i have?" )

if st.button("Find my Answer"):
  if question:
    with st.spinner("Generate response.."):
      st.session_state.messages.append({"role" : "user", "content" : question})
      st.chat_message("user").write(question)

    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    response = assistant_agent.run(st.session_state.messages, callbacks= [st_cb])

    st.session_state.messages.append({'role' : 'assistant', "content": response})
    st.write('### Response')
    st.success(response)

  else:

    st.warning("Please enter the question")    


