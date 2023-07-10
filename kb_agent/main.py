import pinecone
import openai
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.utilities import SerpAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.prompts.chat import SystemMessagePromptTemplate


from bs4 import BeautifulSoup
import requests

import streamlit as st
from streamlit_chat import message
# from utils import *


###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)


def open_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

###     search operations



class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        import re
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            import re
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()

            return stripped_text

        stripped_content = strip_html_tags(html_content)
        # replace double newlines and double spaces with a single pace
        pattern=re.compile("([\n\n])|([\ ]{2,})")
        stripped_content = re.sub(pattern,' ',str(stripped_content))

        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content
    
    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")





def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    # top_p=1,
    # temperature=0.7,
    # max_tokens=256,
    top_p=0.5,
    temperature=0.5,
    max_tokens=512,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def main():

    
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")

    st.set_page_config(layout="wide")
    st.title("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

    st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")
    ### credentials
    with st.sidebar:


        openai_api_key = st.text_input("OpenAI API key", type="password")
        
        prefill_pinecone_environment = "{}".format(PINECONE_ENVIRONMENT)
        prefill_pinecone_index = "{}".format(PINECONE_INDEX)

      
        pinecone_api_key = st.text_input("PineCone API Key", type="password")
        pinecone_env = st.text_input("PINECONE_ENVIRONMENT", value=prefill_pinecone_environment)
        pinecone_index = st.text_input("PINECONE_INDEX", value=prefill_pinecone_index)
  

    if st.button("Enter credentials") or st.session_state.get("credentials_entered", False):
        # Validate inputs
        if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index:
            st.warning(f"Please provide the missing fields.")

        else:
            openai.api_key = openai_api_key
            # Set flag indicating credentials are entered
            st.session_state["credentials_entered"] = True

            # Chat functionality starts here since all credentials are provided
            # Initialize session state if not already done

            if 'responses' not in st.session_state:
                st.session_state['responses'] = ["Welcome to the large language model knowledge base."]

            if 'requests' not in st.session_state:
                st.session_state['requests'] = []

            page_getter = WebPageTool()


            pinecone.init(api_key=pinecone_api_key, environment=PINECONE_ENVIRONMENT)
            index = pinecone.Index(PINECONE_INDEX)

            def find_match(input,index,top_k=5):
                # input_em = model.encode(input).tolist()
                EMBEDDING_MODEL = "text-embedding-ada-002"
                input_em = openai.Embedding.create(
                                                        input=input,
                                                        model=EMBEDDING_MODEL,
                                                        )["data"][0]['embedding']
                
                result = index.query(input_em, top_k=top_k, includeMetadata=True)
                return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


            GoogleSerper = SerpAPIWrapper(serpapi_api_key=os.environ["SERPER_API_KEY"])
            search = Tool(
                    name="search",
                    func=GoogleSerper.run,
                    description="useful for when you need to answer questions about current or recent events. You should ask targeted questions"
                )



            kb_tool = Tool(
                name='kb_tool',
                func= find_match,
                description="Useful for when you need to answer questions about large language models."
            )

            fixed_prompt = '''Assistant is a large language model trained by OpenAI.

            Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

            Assistant doesn't know anything about large language models or anything related to the meaning of life and should use a tool for questions about these topics.

            Assistant also doesn't know information about content on webpages and should always check if asked.

            Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

            tools = [page_getter, kb_tool]

            conversational_agent = initialize_agent(
                agent='chat-conversational-react-description', 
                tools=tools, 
                llm=turbo_llm,
                verbose=True,
                max_iterations=3,
                memory=memory
            )
            conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

            # container for chat history
            response_container = st.container()
            # container for text box
            textcontainer = st.container()


            with textcontainer:
                query = st.text_input("Query: ", key="input")
                if query:
                    with st.spinner("typing..."):
                        conversation_string = get_conversation_string()
                        # st.code(conversation_string)
                        refined_query = query_refiner(conversation_string, query)
                        st.subheader("Refined Query:")
                        st.write(refined_query)
                        context = find_match(refined_query,index)
                        # print(context)  
                        response = conversational_agent.run(refined_query)
                    st.session_state.requests.append(query)
                    st.session_state.responses.append(response) 
            with response_container:
                if st.session_state['responses']:

                    for i in range(len(st.session_state['responses'])):
                        message(st.session_state['responses'][i],key=str(i))
                        if i < len(st.session_state['requests']):
                            message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')



if __name__ == "__main__":
    main()


                    

          