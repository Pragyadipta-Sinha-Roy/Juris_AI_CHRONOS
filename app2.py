import os
from dotenv import load_dotenv
from pyprojroot import here
load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
VECTORDB_DIR = r"C:\Users\KIIT0001\Documents\ML project\JurisAi-Chronos\legal_vectordb"
K = 2

@tool
def lookup_policy(query: str)->str:
    """Check the Constitution of India and the Indian Penal Code documents to answer questions related to it."""
    vectordb = Chroma(
    collection_name="legal_rag-chroma",
    persist_directory=str(here(VECTORDB_DIR)),
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )
    docs = vectordb.similarity_search(query, k=K)
    return "\n\n".join([doc.page_content for doc in docs])

sqldb_directory = here(r"C:\Users\KIIT0001\Documents\ML project\JurisAi-Chronos\db\legal.db")

sql_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOpenAI(model="gpt-4o")
system_role = """You are an agent designed to interact with a Legal Case database. You have access to a database containing case ids, case_outcome, case_title, case_text. Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
    Question: {question}\n
    SQL Query: {query}\n
    SQL Result: {result}\n
    Answer:
    """
db = SQLDatabase.from_uri(
    f"sqlite:///{sqldb_directory}")

execute_query = QuerySQLDatabaseTool(db=db)
write_query = create_sql_query_chain(
    sql_llm, db)
answer_prompt = PromptTemplate.from_template(
    system_role)


answer = answer_prompt | sql_llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

@tool
def query_sqldb(query):
    """
    Executes a SQL query and returns the response.
    """
    response = chain.invoke({"question": query})
    return response

from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=2)

tools = [search_tool,lookup_policy, query_sqldb]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    user_message = state["messages"][-1].content
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
import json
from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[search_tool, lookup_policy, query_sqldb])
graph_builder.add_node("tools", tool_node)
from langgraph.graph import END, MessagesState
from typing import Literal

from langgraph.graph import END, MessagesState
from typing import Literal

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    ["tools", END],
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


import streamlit as st
from langchain_core.messages import HumanMessage

class ChatInterface:
    def __init__(self):
        self.config = {"configurable": {"thread_id": "1"}}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""
    
    def chat(self, message):
        """Process chat messages and maintain history."""
        if not message:
            return
        
        try:
            final_state = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=self.config
            )
            
            bot_response = final_state["messages"][-1].content
            
            st.session_state.chat_history.append((message, bot_response))
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.chat_history.append((message, error_message))
    
    def create_interface(self):
        """Create and configure the Streamlit interface with improved UI."""
        st.set_page_config(page_title="JurisAI Chatbot", page_icon="🤖", layout="wide")
        
        st.markdown("""
        <style>
            .chat-container {
                max-width: 800px;
                margin: auto;
                padding: 20px;
                border-radius: 10px;
                background-color: #ffffff;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            .chat-message {
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
                color: #ffffff;
            }
            .user {
                background-color: #0078D7;
                text-align: left;
            }
            .assistant {
                background-color: #28a745;
                text-align: left;
            }
            .button-container {
                display: flex;
                justify-content: space-between;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='chat-container'>
            <h1 style='text-align: center; color: #0078D7;'>JurisAI Chatbot</h1>
            <p style='text-align: center; color: #333;'>Welcome to our JurisAI chatbot! Ask me anything legal.</p>
        </div>
        """, unsafe_allow_html=True)
        
        chat_container = st.container()
        with chat_container:
            for user_msg, bot_msg in st.session_state.chat_history:
                st.markdown(f"<div class='chat-message user'><b>You:</b> {user_msg}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message assistant'><b>JurisAI:</b> {bot_msg}</div>", unsafe_allow_html=True)
        
        user_input = st.text_input("Type your message here...", value=st.session_state.user_input)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Send", use_container_width=True):
                self.chat(user_input)
                st.session_state.user_input = ""  # Clear input field in session state
                st.experimental_rerun()
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.experimental_rerun()

if __name__ == "__main__":
    chat_app = ChatInterface()
    chat_app.create_interface()

