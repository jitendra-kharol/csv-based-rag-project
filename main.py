from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma

emb_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

vectorstore = Chroma(
    persist_directory="Chroma-DB",
    embedding_function= emb_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "lambda_mult": 1
    }
)

'''retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)'''

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the user's question using only the information provided in the context.

If the answer is not available in the context, say:
"I could not find the answer in the provided data."

Context:
{context}

 query:
{query}

Answer:
""")

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b"
    )
)

print("______ASK ANY QUESTION BASED ON YOUR  CSV FILE_______")
print("Type Exit To Quit")
while True:
    query = input("you-:")
    if query.lower() == "exit":
        print("Bye")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
            [doc.page_content for doc in docs]
        )
        
    final_prompt = prompt.invoke({
            "context" :context,
            "query": query
        })
        
    response = model.invoke(final_prompt)
    
    print(f"\n AI: {response.content}")
