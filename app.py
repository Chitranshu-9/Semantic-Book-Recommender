import pandas as pd
import numpy as np


# from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# replaced with open AI Embeddings 
from langchain_huggingface import HuggingFaceEmbeddings


import gradio as gr

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    'our-cover.jpg',
    books["large_thumbnail"]
)

# raw_documents = TextLoader("tagged_description.txt", encoding="utf-8",autodetect_encoding=True).load()
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)

documents = []

with open(
    "tagged_description.txt",
    encoding="utf-8",
    errors="ignore"
) as f:
    for line in f:
        line = line.strip()
        if line:
            documents.append(
                Document(page_content=line)
            )


# free embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)

def retrieve_semantic_recommendations(query:str, category:str = None, tone:str=None, initial_top_k: int = 50,final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list=[int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs= books[books['isbn13'].isin(books_list)].head(initial_top_k)

    if category != "All":
        books_recs=books_recs[books_recs["simple_categories"]== category].head(final_top_k)
    else:
        books_recs=books_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

print("working")