# CSV Based RAG

This is a simple CSV-based RAG project I created while practicing LangChain, embeddings, vector databases, and Hugging Face LLMs.

The project uses a sales dataset (`sales_dataset.csv`) and allows you to ask questions about the data.

## Project Structure

```text
csv-based-rag-project/
│
├── csv_files/
│   └── sales_dataset.csv
│
├── create_database.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

First, `create_database.py` reads the CSV file, creates embeddings, and stores them in ChromaDB.

Then, `main.py` uses the vector store, retriever, and Hugging Face LLM to answer questions from the CSV data.

## Dataset

The `sales_dataset.csv` file is available inside the `csv_files` folder.

If you download this project from GitHub, make sure the CSV file is present in:

```text
csv_files/sales_dataset.csv
```

## How to Run

First install the required packages:

```bash
pip install -r requirements.txt
```

Add your Hugging Face API key in the `.env` file.

Then create the vector database:

```bash
python create_database.py
```

After that, run:

```bash
python main.py
```

Now you can ask questions about the sales dataset.

## Technologies Used

* Python
* LangChain
* Hugging Face
* ChromaDB
* Embeddings
* RAG
* Pandas
