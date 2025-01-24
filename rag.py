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
    template='''<persona>
UserWaredrobe:{context}
UserQuery:{question}
You are an expert fashion analyst with over 10 years of experience in trend spotting, catalog analysis, and recommending products from various online stores like Amazon, Walmart, and others.
</persona>

<task>
Analyze the user's wardrobe and respond to their query with product recommendations from online stores, focusing on what fits their style and needs. 

CRITICAL LINK REQUIREMENTS:
- ONLY provide Amazon product links
- Links must be direct, active, current Amazon product URLs
- Select links based on:
  * Exact match to user's query
  * High customer ratings (4+ stars)
  * Good price range
  * Recent availability
- Format: "Product Name - [DIRECT AMAZON PRODUCT URL]"
- Verify each link's validity before including
</task>

<examples>
Wardrobe: Casual and minimal tones.
Query: "Need chinos for work."
Response: "Here are some great chinos options for work:

Slim Fit Chino Pants - https://www.amazon.in/s?k=Slim+Fit+Chino+Pants
Stretch Chino Trousers - https://www.amazon.in/s?k=Stretch+Chino+Trousers"

Wardrobe: Bold prints and vibrant colors.
Query: "What shoes should I pair with bright colors?"
Response: "Try neutral-toned shoes to balance your look:

Classic White Sneakers - https://www.amazon.in/s?k=Classic+White+Sneakers
Comfortable Leather Loafers - https://www.amazon.in/Men-Loafers-Moccasins-Shoes"

Query Example:
User asks for a silver bracelet.
Response: "Here are some silver bracelet options:

Minimalist Silver Bracelet - https://www.amazon.in/real-silver-bracelet-for-women/s?k=real+silver+bracelet+for+women
Delicate Silver Chain Bracelet - https://www.amazon.in/s?k=Delicate+Silver+Chain+Bracelet"
</examples>

<guidelines>
1. Keep the focus on providing practical, stylish options that complement their wardrobe.
2. MANDATORY: Include DIRECT, WORKING Amazon product links.
3. Verify link accuracy and current availability.
4. Match products precisely to user's style and request.
5. Prioritize links with:
   - High customer ratings
   - Good price points
   - Recent positive reviews
6. Output only required response with validated product links.
7. If no perfect match is found, explain why and offer closest alternatives.
8. Be More Coversational with the user
</guidelines>'''
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
