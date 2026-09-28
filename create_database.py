from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

data = CSVLoader(
    file_path="D:/csv-based rag project/csv_files/sales_dataset.csv",
     encoding="utf-8-sig"
     )
     
docs = data.load()

emb_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= emb_model,
    persist_directory="Chroma-DB"

)