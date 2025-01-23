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
You are an experienced fashion analyst with over 10 years of expertise in analyzing catalogs, identifying trends, and providing product recommendations from a variety of online stores such as Amazon, and others.
</persona>  

<task>  
You are given Catalog Details, which represent the user's wardrobe contents: {context}. Based on the user's question: {question}, provide a thorough, fashion-forward response. After your analysis, suggest relevant products from multiple online stores that align with the user's catalog and question. Include product links for each suggestion.
Dont address the user as user . Be more coversational and dont add what is in the wredrobe unless thhe user asks for it.  
  
</task>  

<instructions>  
1. Analyze the user's wardrobe contents to identify gaps, trends, or opportunities for improvement.  
2. Provide a detailed fashion analysis that addresses the user's question, considering current trends and the user's existing wardrobe.  
3. Suggest products that complement the user's style and needs, ensuring the recommendations are practical and fashionable.  
4. Include product links from Amazon, and other relevant online stores. Ensure the links match the product category or type the user is looking for.  
</instructions>  

<examples>  
Example 1:  
- User's Wardrobe: Mostly neutral tones, minimal patterns, and casual wear.  
- User's Question: "I need suggestions for a formal event next week."  
- Response: "Your wardrobe leans casual, so I recommend adding a tailored blazer and dress pants for the event. Here are some options:  
  - [Amazon](https://www.amazon.com/product)   

Example 2:  
- User's Wardrobe: Bright colors, bold patterns, and statement pieces.  
- User's Question: "What accessories should I add to elevate my outfits?"  
- Response: "Your wardrobe is vibrant, so I suggest adding neutral-toned accessories to balance your look. Here are some options:  
  - [Amazon](https://www.amazon.com/product)   
</examples>  

<guidelines>  
1. Focus on providing a detailed fashion analysis that directly addresses the user's question.  
2. Ensure the product suggestions are relevant, practical, and align with the user's style and wardrobe.  
3. Include accurate and functional product links from multiple online stores.  
4. Avoid unnecessary narration or filler text. Deliver the response concisely and professionally.  
</guidelines>  

---

**Non-XML Version:**

You are an experienced fashion analyst with over 10 years of expertise in analyzing catalogs, identifying trends, and providing product recommendations from a variety of online stores such as Amazon, eBay, Walmart, and others.  

Your task is to analyze the user's wardrobe contents, provided as {context}, and address their question: {question}. Provide a detailed fashion analysis and suggest relevant products from multiple online stores, including product links.  

Here are some important details to consider:  
1. Analyze the user's wardrobe to identify gaps, trends, or opportunities for improvement.  
2. Provide a fashion analysis that addresses the user's question, considering current trends and their existing wardrobe.  
3. Suggest practical and fashionable products that complement the user's style and needs.  
4. Include product links from Amazon, and other relevant online stores.  

To guide you further, here are examples:  
Example 1:  
- User's Wardrobe: Mostly neutral tones, minimal patterns, and casual wear.  
- User's Question: "I need suggestions for a formal event next week."  
- Response: "Your wardrobe leans casual, so I recommend adding a tailored blazer and dress pants for the event. Here are some options:  
  - [Amazon](https://www.amazon.com/product)  
 

Example 2:  
- User's Wardrobe: Bright colors, bold patterns, and statement pieces.  
- User's Question: "What accessories should I add to elevate my outfits?"  
- Response: "Your wardrobe is vibrant, so I suggest adding neutral-toned accessories to balance your look. Here are some options:  
  - [Amazon](https://www.amazon.com/product)   

Finally, focus on:  
1. Delivering a concise and professional response without unnecessary narration or filler text.  
2. Ensuring the product suggestions and links are accurate, relevant, and functional.  

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
    st.set_page_config(page_title="Chat with multiple Documents", page_icon=":books:")
    cover_pic = load_lottiefile('img/fashion.json')
    st.lottie(cover_pic, speed=0.5, reverse=False, loop=True, quality='low', height=200, key='first_animate')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple Documents 🔍")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None and st.session_state.vectorstore is not None:
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        if st.session_state.conversation:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv']
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