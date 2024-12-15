from langgraph.graph import MessagesState,StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage,AnyMessage
from langgraph.prebuilt import tools_condition # this is the checker for the if you got a tool back
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os
from langchain_openai import ChatOpenAI
import requests

from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o')
def query_rephraser_sunna(user_query: str) -> List[str]:
    """
    Rephrase and decompose a user query into smaller, more informative sub-queries
    for effective retrieval augmented generation (RAG) processing.

    This function uses a Language Learning Model (LLM) to analyze the input query and 
    generate a structured list of questions tailored for RAG, which relies on retrieving 
    relevant chunks of information from a vector database. The generated questions are 
    designed to enhance retrieval accuracy and cover all aspects of the user's input query.

    Args:
        user_query (str): The original user query that may be complex or ambiguous.

    Returns:
        List[str]: A list of rephrased and decomposed questions suitable for RAG processing.

    Example:
        Input: "Give me KPI of last four years."
        Output: ["KPI of 2021", "KPI of 2022", "KPI of 2023", "KPI of 2024"]

    Notes:
        - The function assumes the presence of an initialized `llm` instance with structured 
          output capabilities.
        - It uses a chat prompt template to instruct the LLM on question generation rules.
    """

    class RAGQuestions(BaseModel):
        query_question: List[str] = Field(
            description="Meaning full questions to get data from RAG"
        )
        rephrased_query: str = Field(description="The rephrased query for better understanding and reterival.")

    # Configuring LLM for structured output
    llm_structure = llm.with_structured_output(RAGQuestions)

    # Define the prompt for the assistant
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are part of Query Simplify Shop Assistant.  Shop Assistant relies on RAG, which cannot handle complex queries directly. "
                    "Your task is to break down complex queries into smaller, more specific questions for accurate retrieval. "
                    "RAG uses retrieval-augmented generation, fetching relevant chunks based on similarity from a vector database "
                    "and answering queries based on those facts. Generate multiple sub-questions if necessary, making them "
                    "informative and context-rich to ensure precise retrieval. Structure your response as a Python list of questions, "
                    "and avoid any extra information or preambles. "
                    "Do not generate more than 5 questions."
                    "Generate More question if and only if required. Like user ask something difficult"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Build the processing chain
    chain = prompt | llm_structure

    # Invoke the LLM to process the user query
    return chain.invoke([user_query])




def make_post_request(url, payload, headers=None):
    """
    Makes a POST request to the given URL with the specified payload and headers.
    
    Parameters:
    - url (str): The API endpoint URL.
    - payload (dict): The data to send in the POST request.
    - headers (dict): Optional headers for the request.
    
    Returns:
    - dict: The JSON response from the API.
    """
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def query_api_sunna(query, top_k=5):
    """
    Queries the API with a specified query and number of top results.
    
    Parameters:
    - query (str): The search query.
    - top_k (int): The number of top results to retrieve.
    
    Returns:
    - dict: The JSON response containing the documents.
    """
    url = "http://0.0.0.0:9062/search_musk"
    payload = {
        "query": query,
        "top_k": top_k
    }
    headers = {
        "Content-Type": "application/json"
    }
    return make_post_request(url, payload, headers)


def flatten_list(nested_list):
    """
    Flattens a nested list into a single list of strings.
    
    Parameters:
    - nested_list (list): A nested list of elements.
    
    Returns:
    - list: A flat list of strings.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursive call for sublist
        else:
            flat_list.append(str(item))  # Convert non-list items to string
    return flat_list

def nested_list_to_string(nested_list, separator=""):
    """
    Converts a nested list into a single concatenated string.
    
    Parameters:
    - nested_list (list): A nested list of elements.
    - separator (str): Separator to use between concatenated strings.
    
    Returns:
    - str: Concatenated string from the nested list.
    """
    flat_list = flatten_list(nested_list)
    return separator.join(flat_list)


from typing import List

def rag_agent_suna(query: str) -> str:
    """
    A RAG (Retrieval-Augmented Generation) agent that takes a query, retrieves relevant documents,
    and generates a concise answer using a conversational model.

    Parameters:
    - query (str): The user's query.

    Returns:
    - str: The model's generated response or an error message if something goes wrong.
    """
    # Define a system prompt for the model
    system_prompt: str = """You are a product recommendation agent for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise. You will be used in a conversational voice system, so ensure clarity.
    Context: {context}:
    
    Provide Response in such a way it should be conversational. Do not add markup or formating. A plain text. 
    """

    try:
        # Step 1: Rephrase the query
        question_rephraser_output: Any = query_api_sunna(query)  # Replace Any with the actual type returned by query_rephraser
        rephrased_questions: List[str] = question_rephraser_output.query_question

        # Step 2: Retrieve relevant documents for each rephrased question
        retrieved_documents: List[dict] = [query_api_sunna(question) for question in rephrased_questions]

        # Extract and concatenate the retrieved documents
        docs: List[List[str]] = [doc_data.get('documents', []) for doc_data in retrieved_documents]
        docs_text: str = nested_list_to_string(docs, separator=" ")

        # Step 3: Format the system prompt with the retrieved context
        formatted_system_prompt: str = system_prompt.format(context=docs_text)

        # Step 4: Create a conversational model instance
        model: ChatOpenAI = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.3)

        # Step 5: Generate a response
        response = model.invoke([
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=query)
        ])
        
        # Ensure the response content is always a string
        if hasattr(response, "content") and isinstance(response.content, str):
            return response.content.strip()
        else:
            return "The model did not provide a valid response."
    except Exception as e:
        # Catch and handle all errors, returning an error message as a string
        return f"An error occurred while processing the request: {str(e)}"


tools = [rag_agent]

llm_with_tools = llm.bind_tools(tools=tools)

def reasoner(state):
    query = state["query"]
    messages = state["messages"]
    # System message
    sys_msg = SystemMessage(content="You are finance expert at Nagros named CFO 2. You will help user with Finance and Company affairs. You are backed with company data tools named rag_agent. Please response in plain text because it will be pass to Voice agent. Do not add markdown etc.")
    message = HumanMessage(content=query)
    messages.append(message)
    result = [llm_with_tools.invoke([sys_msg] + messages)]
    return {"messages":result}

class GraphState(TypedDict):
    """State of the graph."""
    query: str
    final_answer: str
    messages: Annotated[list[AnyMessage], operator.add]


# Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tools)) # for the tools

# Add Edges
workflow.add_edge(START, "reasoner")

workflow.add_conditional_edges(
    "reasoner",
    # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
    # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
    tools_condition,
)
workflow.add_edge("tools", "reasoner")
react_graph = workflow.compile()




