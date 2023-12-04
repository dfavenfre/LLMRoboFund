from Helpers.helper_functions import filter_embeddings
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_sql_agent
from langchain.chat_models import ChatOpenAI, ChatCohere
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.sql_database import SQLDatabase
from langchain.chains import RetrievalQA
from langchain.agents import AgentType
from dotenv import load_dotenv
import pinecone
import json
import os

load_dotenv()


def query_from_sql():
    """
    Use this tool when you need to fetch some data from SQL database to answer\
    questions related to monthly and annual returns of funds,\
    management fees, number of initial outstanding shares, and etc.
    """
    sql_prompt = ""
    with open("Prompts/sql_agent_prompts.json", "r") as file:
        sql_prompt = json.load(file)

    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=1e-10,
        model="gpt-3.5-turbo",
    )

    database = SQLDatabase.from_uri(database_uri=os.environ.get("uri_path"))
    toolkit = SQLDatabaseToolkit(
        db=database,
        llm=llm,
    )
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        early_stopping_method="force",
        suffix=str(sql_prompt),
        max_iterations=10,
        verbose=True,
    )

    return sql_agent


def query_from_vdb(text: str) -> str:
    """
    Use this tool when you need to answer questions related to a funds' financial risks,\
    investment strategy and the invested financial instruments.\
    Also you can answer general information about funds, such as the managing company.
    """

    chat_model = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-3.5-turbo",
    )
    embeddings_cohere = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-ada-002"
    )
    pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),
        environment=os.environ.get("pinecone_environment_value"),
    )

    searcher = Pinecone.from_existing_index(
        index_name=os.environ.get("pinecone_index"), embedding=embeddings_cohere
    )
    context_compressor_retriever = filter_embeddings(
        search_object=searcher, embedding_model=embeddings_cohere
    )

    chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=context_compressor_retriever,
    )
    return chain.run(text)
