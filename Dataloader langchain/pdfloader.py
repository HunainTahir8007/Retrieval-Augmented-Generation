from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path=r"C:\Users\Hunain Xps\Desktop\Dataloader langchain\Introduction_to_AI_and_ML.pdf"
)

docs = loader.load()

print(docs[0].page_content)