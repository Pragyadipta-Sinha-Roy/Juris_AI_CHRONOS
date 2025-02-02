import os
import yaml
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine


class PrepareVectorDB:

    def __init__(self, file_path, chunk_size, chunk_overlap, embedding_model, vectordb_dir, collection_name):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name


    def validate_file(self) -> bool:
        """
        Validates if the provided file exists and is a PDF.

        Returns:
            bool: True if file exists and is a PDF, False otherwise
        """
        if not os.path.exists(here(self.file_path)):
            print(f"Error: File '{self.file_path}' does not exist.")
            return False
        
        if not self.file_path.lower().endswith('.pdf'):
            print(f"Error: File '{self.file_path}' is not a PDF file.")
            return False
        
        return True

    def run(self) -> None:
        """
        Executes the main logic to create and store document embeddings in a VectorDB.

        If the vector database directory doesn't exist:
        - It loads the PDF document from file_path, splits it into chunks
        - Embeds the document chunks using the specified embedding model
        - Stores the embeddings in a persistent VectorDB directory

        If the directory already exists, it skips the embedding creation process.

        Returns:
            None
        """
        # Validate input file
        if not self.validate_file():
            return

        if not os.path.exists(here(self.vectordb_dir)):
            # Create the directory and embeddings
            os.makedirs(here(self.vectordb_dir))
            print(f"Directory '{self.vectordb_dir}' was created.")

            # Load and split the single PDF file
            loader = PyPDFLoader(str(here(self.file_path)))
            docs_list = loader.load_and_split()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Add to vectorDB
            vectordb = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=OpenAIEmbeddings(model=self.embedding_model),
                persist_directory=str(here(self.vectordb_dir))
            )
            print("VectorDB is created and saved.")
            print(f"Number of vectors in vectordb: {vectordb._collection.count()}\n")
        else:
            print(f"Directory '{self.vectordb_dir}' already exists.")


if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")

    prepare_db_instance = PrepareVectorDB(
        file_path=r"C:\Users\KIIT0001\Documents\ML project\JurisAi-Chronos\ilovepdf_merged.pdf",
        chunk_size=10000,
        chunk_overlap=1000,
        embedding_model='text-embedding-3-small',
        vectordb_dir='legal_vectordb',
        collection_name='legal_rag-chroma'
    )
    
    prepare_db_instance.run()


# Database setup
# database_file_path = "./db/legal.db"
# engine = create_engine(f"sqlite:///{database_file_path}")
# file_url = r"C:\Users\KIIT0001\Documents\ML project\JurisAi-Chronos\case_dataset.xlsx" 
# os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
# df = pd.read_excel(file_url).fillna(value=0)
# df.to_sql("Legal_case_details", con=engine, if_exists="replace", index=False)


