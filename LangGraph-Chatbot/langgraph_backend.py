from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGroq(model='qwen/qwen3.8-27b', max_tokens=5000)

# define state 
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define functions 
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# define graph 
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)