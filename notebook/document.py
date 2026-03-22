# Document Structure

from langchain_core.documents import Document
import os

# doc = Document(
#     page_content="Main text content i am using to create RAG.",
#     metadata = {
#         "source":"example",
#         "Pages": 1,
#         "author": "None",
#         "date_created":"1999"
#     }
# )



# Text Loader

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data\\text_files\\python_intro.txt", encoding="utf-8")
document = loader.load()
# print(document)



# Directory Loader

from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(
    "data\\text_files",
    glob="**/*.txt",
    loader_cls= TextLoader,
    loader_kwargs={'encoding':'utf-8'},
    show_progress=False
)

documents = dir_loader.load()
# print(documents)



# Pdf Loader

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

dir_loader = DirectoryLoader(
    "data\\pdf",
    glob="**/*.pdf",
    loader_cls= PyMuPDFLoader,
    show_progress=False
)

documents_pdf = dir_loader.load()
# print(documents_pdf)

