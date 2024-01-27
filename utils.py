#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.llms import OpenAI
import streamlit as st
#import plotly.express as px

try:
    from langchain_community.graphs.memgraph_graph import RAW_SCHEMA_QUERY
except ImportError:
    RAW_SCHEMA_QUERY = None  # Provide a placeholder if missing

def query_agent(data, query):

    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    # llm = OpenAI()
    llm = OpenAI(temperature=0)
    
    # Create a Pandas DataFrame agent.
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    #Python REPL: A Python shell used to evaluating and executing Python commands. 
    #It takes python code as input and outputs the result. The input python code can be generated from another tool in the LangChain
    
    return agent.run(query)

def table_agent(data, query):
    """Generates a table response using a Pandas DataFrame agent."""
    try:
        df = pd.read_csv(data)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df,verbose=False)
        # table = agent.run(query)
        # return table  # Return the generated table
        table_prompt = f"Create a table that answers the following question about the dataset:\n{query}"  # Explicit table prompt
        table_str = agent.run(table_prompt)
        rows = table_str.splitlines()
    
        if rows:  # Check if any rows exist
            first_line = rows[0]  # Store the first line
            data = [row.split() for row in rows[1:] if row.strip()]  # Process remaining rows
    
            if data:
                columns = data[0]
                data = data[1:]
                df = pd.DataFrame(data, columns=columns)
                return first_line, df  # Return first line and DataFrame
            else:
                return first_line, None  # Return first line and None if no table data
        else:
            return None, None  # Return None for both if no response
    except Exception as e:
        st.error("Error generating table:", e)
        return None

def graph_agent(data, query):
    """Generates a graph response using the LLM."""
    try:
        llm = OpenAI()
        response = llm(query)  # Prompt the LLM for graph generation
        data_for_chart = extract_data_for_chart_from_response(response)  # Extract data from LLM response
        fig = px.bar(data_for_chart)  # Adjust chart type as needed
        return fig  # Return the generated figure
    except Exception as e:
        st.error("Error generating graph:", e)
        return None    
