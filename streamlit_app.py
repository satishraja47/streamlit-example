import streamlit as st
from dotenv import load_dotenv
from utils import *

load_dotenv()


st.title("Let's do some analysis on your CSV")
st.header("Please upload your CSV file here:")

# Capture the CSV file
data = st.file_uploader("Upload CSV file",type="csv")

query = st.text_area("Enter your query")

# Determine desired format
desired_format = st.radio("Choose output format:", ["Text", "Table", "Graph"])

button = st.button("Generate Response")

if button:
        # Generate response with appropriate agent
        if desired_format == "Table":
            response = table_agent(data,query) # Use LLM for table generation
            st.table(response)
        elif desired_format == "Graph":
            response = graph_agent(data,query)  # Use LLM for graph generation
            st.plotly_chart(response)
        else:
            response = query_agent(data,query)  # Default to text response
            st.write(response)

    
