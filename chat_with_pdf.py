import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# 오픈 api 키를 넣을때는 변수에다가 넣는게 아니고 터미널에다가 set OPENAI_API_KEY= 따음표 없이 openai api 키를 넣는다. 여기가 제일 어렵고 힘들었다.



DOC_PATH = "pdf 파일 경로"
CHROMA_PATH = "벡터 DB가 담긴 경로" 



#pdf 파일 넣고 업로드한다. 여기서는 회사 넥스트랩을 넣고 업로드한다.
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

# 청크로 나눈다. 청크 하나당 크기가 여기서는 500자이다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# 오픈 api 를 이용해 임베딩한다.
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 벡터 임베딩값을 chroma 벡터 데이터베이스에 넣는다.
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)


# 질문한다.
query = input('넥스트랩에 대해서 질문하세요 고객님 \n\n')

#제일 벡터 값이 가까운 3개의 청크를 고려한다.
docs_chroma = db_chroma.similarity_search_with_score(query, k=3)

# 사용자 질문과 검색된 문맥 정보를 기반으로 답변 생성
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

# you can use a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don't justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# 템플렛을 바탕을 질문과 답변을 해라
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# llm 모델을 불러와 질문과 답변한다.
model = ChatOpenAI()
response_text = model.invoke(prompt)


#질문에 대한 답변이다.
print(f'\n{response_text.content}')

#답변의 근거가 되는 페이지다.
print(f"\n{docs_chroma[0][0].metadata['page']+1} 페이지에 근거가 있다.")
