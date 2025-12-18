# import pandas as pd
# import numpy as np


# # from langchain_community.document_loaders import TextLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_chroma import Chroma

# # replaced with open AI Embeddings 
# from langchain_huggingface import HuggingFaceEmbeddings


# import gradio as gr

# books = pd.read_csv("books_with_emotions.csv")
# books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# books["large_thumbnail"] = np.where(
#     books["large_thumbnail"].isna(),
#     'our-cover.jpg',
#     books["large_thumbnail"]
# )

# # raw_documents = TextLoader("tagged_description.txt", encoding="utf-8",autodetect_encoding=True).load()
# # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
# # documents = text_splitter.split_documents(raw_documents)

# documents = []

# with open(
#     "tagged_description.txt",
#     encoding="utf-8",
#     errors="ignore"
# ) as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             documents.append(
#                 Document(page_content=line)
#             )


# # free embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_books = Chroma.from_documents(
#     documents,
#     embedding=embeddings
# )

# def retrieve_semantic_recommendations(query:str, category:str = None, tone:str=None, initial_top_k: int = 50,final_top_k: int = 16) -> pd.DataFrame:
#     recs = db_books.similarity_search(query, k=initial_top_k)
#     books_list=[int(rec.page_content.strip('"').split()[0]) for rec in recs]
#     books_recs= books[books['isbn13'].isin(books_list)].head(initial_top_k)

#     if category != "All":
#         books_recs=books_recs[books_recs["simple_categories"]== category].head(final_top_k)
#     else:
#         books_recs=books_recs.head(final_top_k)

#     if tone == "Happy":
#         book_recs.sort_values(by="joy", ascending=False, inplace=True)
#     elif tone == "Surprising":
#         book_recs.sort_values(by="surprise", ascending=False, inplace=True)
#     elif tone == "Angry":
#         book_recs.sort_values(by="anger", ascending=False, inplace=True)
#     elif tone == "Suspenseful":
#         book_recs.sort_values(by="fear", ascending=False, inplace=True)
#     elif tone == "Sad":
#         book_recs.sort_values(by="sadness", ascending=False, inplace=True)

#     book_recs = books.copy()
#     return book_recs

# def recommend_books(query:str, category:str,tone:str):
#     recommendations = retrieve_semantic_recommendations(query,category,tone)
#     results = []

#     for _, row in recommendations.iterrows():
#         description = row['description']
#         truncated_description_split = description.split()
#         truncated_description = " ".join(truncated_description_split[:30]) + "..."

#         authors_split = row['authors'].split(';')
#         if len(authors_split) == 2:
#             authors_str = f"{authors_split[0]} and {authors_split[1]}"
#         elif len(authors_split) > 2:
#             authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
#         else:
#             authors_str = row["authors"]
        
#         caption = f"{row['title']} by {authors_str} : {truncated_description}"

#         results.append((row['large_thumbnail'],caption))
#     return results

# categories = ["All"] + sorted(books['simple_categories'].unique())
# tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# with gr.Blocks(theme = gr.themes.Glass()) as Dashboard:
#     gr.Markdown('# Semantic Book Recommender')

#     with gr.Row():
#         user_query = gr.Textbox(label="Please enter a description of a book:",
#                                 placeholder="e.g., A story about forgiveness")

#         category_dropdown = gr.Dropdown(choices=categories, label="Select a Category:", value="All")
#         tone_description = gr.Dropdown(choices=tones, label="Select an Emotional Tone:", value="All")
#         submit_button = gr.Button("Find Recommendations")
    
#     gr.Markdown('# Recommendations')
#     output = gr.Gallery(label="Recommended Books", columns=8,rows=2)

#     submit_button.click(fn=recommend_books, inputs = [user_query,category_dropdown,tone_description], outputs = output)

# if __name__ == "__main__":
#     Dashboard.launch()
# print("working")


import pandas as pd
import numpy as np

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr

# =========================
# Load and preprocess data
# =========================

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "our-cover.jpg",
    books["large_thumbnail"]
)

# =========================
# Load tagged descriptions
# =========================

documents = []
with open("tagged_description.txt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line:
            documents.append(Document(page_content=line))

# =========================
# Create embeddings + vector DB
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)

# =========================
# Semantic retrieval logic
# =========================

def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:

    # Semantic search
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract ISBNs from tagged descriptions
    books_list = [
        int(rec.page_content.strip('"').split()[0])
        for rec in recs
        if rec.page_content.strip().split()[0].isdigit()
    ]

    # Filter books dataframe
    book_recs = books[books["isbn13"].isin(books_list)].copy()

    # Category filtering
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Tone-based sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)

# =========================
# Formatting for Gradio
# =========================

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"])
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_split[0]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# =========================
# Gradio UI
# =========================

categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as Dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a Category:",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an Emotional Tone:",
            value="All"
        )

        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("# Recommendations")

    output = gr.Gallery(
        label="Recommended Books",
        columns=8,
        rows=2
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    Dashboard.launch(theme=gr.themes.Glass())
