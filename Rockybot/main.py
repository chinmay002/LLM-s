import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


load_dotenv()

st.title('News Research tool')

st.sidebar.title('News article URLS')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button('Process URLs')    
file_path = "vector_db.pkl"

main_placeolder = st.empty()  #ui element for progrss bar


if process_url_clicked:
     #losd the url
     loader = UnstructuredURLLoader(urls = urls)
     data = loader.load()
     main_placeolder.text('data loading is completed')
     #split the conetnts of docs
     splitter = RecursiveCharacterTextSplitter(
          separators= ['\n\n','\n','.',','],
          chunk_size = 1000
     )
     docs = splitter.split_documents(data)
     main_placeolder.text('docs splitting is completed')

     #create emebeding
     embeddings = OpenAIEmbeddings()
     vectorindex = FAISS.from_documents(docs,embeddings)
     main_placeolder.text('Embedding is  completed')

     with open(file_path ,'wb') as f:
          pickle.dump(vectorindex,f)


query =   main_placeolder.text_input('Questions: ')
if query :
     if os.path.exists(file_path):
          with open(file_path,'rb') as f:
               vectorstore = pickle.load(f)
               chain = RetrievalQAWithSourcesChain.from_llm(llm = OpenAI(temperature=0.7),retriever=vectorstore.as_retriever())
               result = chain({'question':query},return_only_outputs = True)
               st.header('Answer')
               st.subheader(result['answer'])


               sources = result.get('sources',"")
               if sources:
                    st.subheader('Sources:')
                    sources_list = sources.split('\n')
                    for source in sources_list:
                         st.write(source)

          