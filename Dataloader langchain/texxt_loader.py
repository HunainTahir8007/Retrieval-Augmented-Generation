import sys
from langchain_community.document_loaders import TextLoader , WebBaseLoader
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.stdout.reconfigure(encoding="utf-8")
from langchain_core.prompts import PromptTemplate

loader = WebBaseLoader(web_path="https://gcuf.edu.pk/about/history/")

docs = loader.load()

load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite" , temperature = 0.8)

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["context"] , 
    template="tell me about the school era in gcuf in deatail in structure format  strict rule give names in numbering  \n{context}"
)


chain = prompt | model | parser

response = chain.invoke({"context": docs[0].page_content})
print(response)