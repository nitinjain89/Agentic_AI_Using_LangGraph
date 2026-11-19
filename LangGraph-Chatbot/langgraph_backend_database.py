from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import sqlite3

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

# Create SQLite connection and initialize checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

# CONFIG = {'configurable': {'thread_id':'thread-1'}}

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
