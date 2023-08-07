import os
import pinecone
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain 
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.callbacks import FileCallbackHandler


st.set_page_config(page_title="KejlGPT", page_icon=":owl:")
st.header(":owl: :books:  :test_tube:   KejlGPT")
st.subheader("Buchi's AI Kjeldahl and Steam Distillation expert!")



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Citations")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)



#define embedding model
embed = OpenAIEmbeddings(openai_api_key="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")

# initialize connection to pinecone
index_name = 'kjex-eddy'
pinecone.init(
    api_key="b9107246-30cc-4e36-ad28-18b97d90ca63",  # app.pinecone.io (console)
    environment="eu-west4-gcp"  # next to API key in console
)
index = pinecone.Index(index_name)

# Define retriever
text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory =ConversationBufferWindowMemory(memory_key="chat_history",chat_memory=msgs, return_messages=True)


# Setup LLM and QA chain
system_prompt="you are a  helpful AI assistant with an expertise in Extraction that explains things in simple terms. . dont forget to keep it short and to the point. please do not engage in nontechnical conversation. "

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key="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", temperature=0.5, streaming=True
)
qa_chain = RetrievalQA.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, 
    )


if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="type your question here!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])