from langchain_community.document_loaders import PyPDFLoader
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#pdf 파일 업로드하고
file_path = "pdf경로"
loader = PyPDFLoader(file_path)

docs = loader.load()

#api 키 불러온다
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#chatgpt 4o 모델 쓰는데 굉장히 똑똑하다
llm = ChatOpenAI(model="gpt-4o")


#텍스트의 용량이 클때 청크로 잘게 나눈다
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#벡터 DB 에 벡터 임베딩해서 담는다
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


#LLM에 답변형식을 정해준다
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

#LLM 기반으로 질문하고 답변한다. pdf에 근거해서
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


query=input('고객님 무엇이든 물어보세요\n\n')

results = rag_chain.invoke({"input": f"{query}"})

#LLM 의 답변 내용이다.
answer=results['answer']
print(f'\n{answer}')

# 어느 페이지에 각각 근거하여 답변했는지를 알려준다.
lst=[]
for i in results["context"]:
    lst.append((i.metadata["page"]+1))

lst.sort()
cv = [f"{j}p" for j in lst]
reason = ", ".join(cv) + " 에 근거한 답변입니다."
print(f'\n{reason}')
