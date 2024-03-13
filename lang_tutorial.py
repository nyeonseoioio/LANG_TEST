from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
import os

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# client = OpenAI(api_key=openai_api_key)

llm = ChatOpenAI(api_key=openai_api_key)
# output = llm.invoke("2024년 청년 지원 정책에 대하여 알려줘")
# print(output)

"""
 ChatPromptTemplate = 대화 형식의 프롬프트를 생성하기 위한 클래스
 from_message 메서드 = 대화의 구조를 설정하는 데 사용 
 prompt template는 위에서 말한 ai가 잘 이해할 수 있는 형태로 만들어주는 prompt engineering을 쉽게 하기 위한 예시 형태들

"""
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 청년을 행복하게 하기 위한 정부정책 안내 컨설턴트야"),
    ("user", "{input}")
])


"""
 pip(|) 연산자를 사용하여 두 가지 작업을 연결
 해당 구조는 사용자의 입력을 대화형 "prompt"로 전달하여 처리하고, 
 그에 대한 결과를 "llm"이라는 AI모델을 사용하여 적절한 응답 생성하는 방식

 """
# chain = prompt | llm 


"""

# invoke 메서드 : 주로 어떤 작업을 호출하거나 실행하는 데 사용, 결과 반환, 프로세스(chain) 실행
# 여기서는 사용자의 실제 입력을 딕셔너리 형태로 전달함. 

"""

# print(chain.invoke({"input" : "2024년 청년 지원 정책에 대해 알려줘"}))




"""
# 내용 파싱하기 


"""

# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser

# print(chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘"}))




"""
# 검색 기능 적용하기

"""

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.moel.go.kr/policy/policyinfo/support/list4.do")

docs = loader.load()

# print(docs)


"""
# embedding : 사람이 이해하는 자연어나 이미지 등의 
복잡한 데이터를 컴퓨터가 처리할 수 있는 숫자 형태의 
벡터로 변환하는 기술

이러한 변환을통해 컴퓨터는 단어, 문장, 이미지 사이의 관계를 수학적으로 계산할 수 있게 됨


1. 의미의 수치화 
  ㄴ '사랑'이나 ' 행복' 같은 추상적인 개념을 컴퓨터가 처리할 수 있는 형태로 바꿀 수 있음

2. 차원 축소
  ㄴ 데이터는 종종 복잡라고, 처리하기 어려운 높은 차원을 가짐, 임베딩은 고차원의 데이터를 저차원으로 효율적으로 표현하여 계산을 용이하게 함

3. 관계파악
  ㄴ 임베딩 벡터는 데이터 사이의 유사성이나 관계를 반영 
  

"""
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# print(embeddings)


from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# 바로 Docs 내용을 반영도 가능합니다.
from langchain_core.documents import Document

document_chain.invoke({
    "input": "국민취업지원제도가 뭐야",
    "context": [Document(page_content="""국민취업지원제도란?

취업을 원하는 사람에게 취업지원서비스를 일괄적으로 제공하고 저소득 구직자에게는 최소한의 소득도 지원하는 한국형 실업부조입니다. 2024년부터 15~69세 저소득층, 청년 등 취업취약계층에게 맞춤형 취업지원서비스와 소득지원을 함께 제공합니다.
[출처] 2024년 달라지는 청년 지원 정책을 확인하세요.|작성자 정부24""")]
})

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "국민취업지원제도가 뭐야"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...