from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.sql_database import SQLDatabase
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI, ChatCohere
from tools.tools import query_from_vdb, query_from_sql
from Helpers.helper_functions import tool_initializer
import json
import os


sql_tool = query_from_sql()


def run_robofund_agent(query: str) -> str:
    system_prompt = ""
    with open("Prompts\intermediate_agent_template.json", "r") as file:
        system_prompt = json.load(file)
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-3.5-turbo",
        model_kwargs={"stop": ["\Observation:"]},
    )

    template = system_prompt[0]["template"]

    tools_for_agent = tool_initializer(
        names=["SearchDocuments", "SearchSQL"],
        functions=[query_from_vdb, sql_tool.run],
        descriptions=[
            """
                Use this tool when you need to answer questions related to a funds financial risks,
                investment strategy and the invested financial instruments.
                Also you can answer general information about funds, such as the managing company.
            """,
            """
                Use this tool when you need to answer questions related to a funds financial risks,
                investment strategy and the invested financial instruments.
                Also you can answer general information about funds.
            """,
        ],
    )

    mrkl = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        
    )

    prompt_template = PromptTemplate(template=template, input_variables=["question"])
    query_result = mrkl.run(prompt_template.format_prompt(question=query))

    return query_result
