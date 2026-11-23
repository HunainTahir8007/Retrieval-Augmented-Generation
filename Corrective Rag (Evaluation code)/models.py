from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


strip_model = ChatGoogleGenerativeAI(model= "gemini-3.1-flash-lite")
model  = ChatGroq(model= "llama-3.3-70b-versatile")
