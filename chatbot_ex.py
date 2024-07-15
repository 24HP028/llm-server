import os
import openai
import sys
import json
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = 'sk-키입력'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

loader = PyPDFLoader("C:/Users/user/Desktop/프로젝트/해상물류/LLMServer/data/TEST.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=tiktoken_len)
texts = text_splitter.split_documents(pages)

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

docsearch = Chroma.from_documents(texts, hf)

openai = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=False, 
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=openai,
    chain_type="stuff",
    retriever=docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10}
    ),
    return_source_documents=True
)

def get_response(input_text):
    return qa.invoke({"query": input_text})

