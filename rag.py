import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import docx2txt
from htmlTemplate import css, bot_template, user_template
import streamlit_lottie as st_lottie
import json


def get_docs_text(docs):
    text = ""
    for doc in docs:
        if doc is not None:
            if doc.type == "text/plain":  # txt doc
                text += str(doc.read(), encoding="utf-8")
            elif doc.type == "application/pdf":  # pdf
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                text += docx2txt.process(doc)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0
    )
    
    # Create a proper prompt template
    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template='''
<persona>  
You are an advanced Amazon product recommendation assistant with expertise in fashion and wardrobe styling. You have 5+ years of experience in analyzing user preferences, understanding wardrobe styles, and providing tailored product suggestions.  
</persona>  

<task>  
Provide Amazon product recommendations matching the user's style and query. Ensure the products are high-rated (4+ stars), reasonably priced, and available. Analyze the user's wardrobe style and provide weather-based and occasion-based suggestions.  
</task>  

<details>  
1. Understand the user's query and wardrobe style context.  
2. Use the provided example links to identify patterns and generate similar product links.  
3. Ensure all recommendations are high-rated (4+ stars), reasonably priced, and currently available.  
4. Provide a summary of the user's wardrobe style and suggest weather-based and occasion-based recommendations.  
</details>  

<examples>  
1. User Query: Chinos for men  
   Wardrobe Style: Casual  
   Example Link: https://www.amazon.in/s?k=chinos+men  
   Recommendation: [Product Name] - [Price] - [Rating] - [Link]  

2. User Query: Chinos for women  
   Wardrobe Style: Office wear  
   Example Link: https://www.amazon.in/s?k=chinos+women  
   Recommendation: [Product Name] - [Price] - [Rating] - [Link]  

3. User Query: Women's dress  
   Wardrobe Style: Party wear  
   Example Link: https://www.amazon.in/s?k=women+wear+dress  
   Recommendation: [Product Name] - [Price] - [Rating] - [Link]  
</examples>  

<guidelines>  
1. Focus on generating accurate and relevant product links based on the user's query and style.  
2. Summarize the user's wardrobe style and provide weather-based (e.g., summer, winter) and occasion-based (e.g., casual, formal, party) suggestions.  
3. Ensure all recommendations meet the criteria of high ratings, reasonable pricing, and availability.  
</guidelines>
    '''
)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory, 
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
    return conversation_chain


def handle_userinput(user_question):

    if st.session_state.conversation is None and st.session_state.vectorstore is not None:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
    
    response = st.session_state.conversation({"question": user_question})

    # Display current question and answer
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", response["answer"]), unsafe_allow_html=True)
    
    # Update session state with the new chat history, avoiding duplicates
    if not st.session_state.chat_history or (
        st.session_state.chat_history[-2].content != user_question and 
        st.session_state.chat_history[-1].content != response["answer"]
    ):
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response["answer"]))

def load_lottiefile(filepath: str):
    '''
    to load a Lottie animation file.
    Takes a file path as input and reads the JSON content of the Lottie file.
    Returns the JSON data for the Lottie animation. 
    '''
    with open(filepath, 'rb') as f:
        return json.load(f)

def main():

    load_dotenv()
    st.set_page_config(page_title="Fashion AI-Assistant", page_icon=":books:")
    cover_pic = load_lottiefile('img/fashion.json')
    st.lottie(cover_pic, speed=0.5, reverse=False, loop=True, quality='low', height=200, key='first_animate')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Fashion AI-Assistant 👘")
    user_question = st.text_input("Ask a question about your Waredrobe:")
    if user_question:
        if st.session_state.conversation is None and st.session_state.vectorstore is not None:
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        if st.session_state.conversation:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Waredrobe")
        pdf_docs = st.file_uploader(
            "Upload your Waredrobe here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv']
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_docs_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                # Create or update vector store
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                else:
                    new_vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore.merge(new_vectorstore)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        if st.button("Chat-History"):
            # Display's Chat History 
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.write(message.content)
                else:
                    st.write(message.content)

if __name__ == "__main__":
    main()
