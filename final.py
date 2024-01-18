#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chromadb
client = chromadb.PersistentClient()


# In[3]:


collection = client.get_or_create_collection(
    name="0117"
)


# In[4]:


import pandas as pd
from tqdm import tqdm

data = pd.read_csv("C:/Users/kt826/Desktop/ECONOQA.csv")
data.sample(5)


# In[5]:


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')


# In[6]:


ids = []
metadatas = []
embeddings = []

for row in tqdm(data.iterrows()):
    index = row[0]
    query = row[1].Question
    answer = row[1].Answer
    
    metadata = {
        "query": query,
        "answer": answer
    }
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    ids.append(str(index))
    metadatas.append(metadata)
    embeddings.append(embedding)
    
chunk_size = 1024  # 한 번에 처리할 chunk 크기 설정
total_chunks = len(embeddings) // chunk_size + 1  # 전체 데이터를 chunk 단위로 나눈 횟수
embeddings = [ e.tolist() for e in tqdm(embeddings)]  

for chunk_idx in tqdm(range(total_chunks)):
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size
    
    # chunk 단위로 데이터 자르기
    chunk_embeddings = embeddings[start_idx:end_idx]
    chunk_ids = ids[start_idx:end_idx]
    chunk_metadatas = metadatas[start_idx:end_idx]
    
    # chunk를 collection에 추가
    collection.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)


# In[7]:


def retriever(question):
    result = collection.query(
        query_embeddings=model.encode(question, normalize_embeddings=True).tolist(),
        n_results=3
    )
    
    prompt = f"""
    You are an intelligent assistant helping the users with their questions on {{company | research papers | …}}. Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.
    The instructions should be in Korean. Reply via text only.

    Do not try to make up an answer:
     - If the answer to the question cannot be determined from the context alone, say "해당 질문은 https://econovation.kr/contact 혹은 에코노베이션 카카오톡 채널 https://pf.kakao.com/_laTLs로 문의주세요!"
     - If the context is empty, just say "I do not know the answer to that."
 
    CONTEXT: 
    {result}
 
    QUESTION:
    {question}
     
    Strictly Use ONLY the following pieces of context to answer the question at the end.
    Helpful Answer:
    """
    return prompt


# In[28]:


import openai
import streamlit as st
import streamlit.components
import streamlit
from streamlit_chat import message

openai.api_key = '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'


# In[29]:


def generate_response(question):
    messages = []
    
    messages.append({"role": "user", "content": question})

    # Assume you have the 'result' variable from the retriever function
    result = retriever(question)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an intelligent assistant."},
                  {"role": "user", "content": question},
                  {"role": "assistant", "content": result}]
    )

    message = completion.choices[0].message['content']
    return message

# In[32]:


import streamlit as st

st.title("ECONOVATION CHATBOT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("말씀해주세요."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Request chat completion 
    response = generate_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# In[ ]:




