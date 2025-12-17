import pandas as pd
import numpy as np


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# replaced with open AI Embeddings 
from langchain_huggingface import HuggingFaceEmbeddings


import gradio as gr

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
print("working")