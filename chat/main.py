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
import streamlit as st
from streamlit_chat import message
# from utils import *





def find_match(input,top_k=5):
    # input_em = model.encode(input).tolist()
    EMBEDDING_MODEL = "text-embedding-ada-002"
    input_em = openai.Embedding.create(
                                            input=input,
                                            model=EMBEDDING_MODEL,
                                            )["data"][0]['embedding']
    
    result = index.query(input_em, top_k=top_k, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="gpt-4-0613",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
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

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")

    st.set_page_config(layout="wide")
    st.title("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

    st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")
    ### credentials
    with st.sidebar:


        openai_api_key = st.text_input("OpenAI API key", type="password")
        prefill_pinecone_api_key = "{}".format(PINECONE_API_KEY)
        prefill_pinecone_environment = "{}".format(PINECONE_ENVIRONMENT)
        prefill_pinecone_index = "{}".format(PINECONE_INDEX)

        st.write(os.getenv("PINECONE_API_KEY"))
        pinecone_api_key = st.text_input("PINECONE_API_KEY", value=prefill_pinecone_api_key)    
        st.write(prefill_pinecone_environment)    
        pinecone_env = st.text_input("PINECONE_ENVIRONMENT", value=prefill_pinecone_environment)
        st.write(prefill_pinecone_index) 
        pinecone_index = st.text_input("PINECONE_INDEX", value=prefill_pinecone_index)

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        index = pinecone.Index(PINECONE_INDEX)
 
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
                st.session_state['responses'] = ["How can I assist you?"]

            if 'requests' not in st.session_state:
                st.session_state['requests'] = []

            llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0.0, top_p=1, frequency_penalty=0, presence_penalty=0, openai_api_key=openai_api_key) 

            if 'buffer_memory' not in st.session_state:
                        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


            system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
            and if the answer is not contained within the text below, say 'I don't know'""")


            human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

            prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

            conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




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
                        context = find_match(refined_query)
                        # print(context)  
                        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
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


                    

          