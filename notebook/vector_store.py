import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import embedding_manager
from document import documents_pdf
from chunks import split_documents
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")

print("Persist directory:", DEFAULT_PERSIST_DIR)
print("Absolute path:", os.path.abspath(DEFAULT_PERSIST_DIR))

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = DEFAULT_PERSIST_DIR):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Create persistent Chromadb client
            os.makedirs(self.persist_directory, exist_ok = True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document for RAG"
                }
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        print(f"Adding {len(documents)} documents to vector store")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embeddings) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embeddings.tolist())


        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            self.client.persist()  

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


vector_store = VectorStore()
# print("Succesfull vectorstore")


# Step 1: split
chunks = split_documents(documents_pdf)

# Step 2: generate embeddings
embeddings = embedding_manager.generate_embeddings(
    [doc.page_content for doc in chunks]
)

print(len(chunks))
print(len(embeddings))

# Step 3: store
vector_store.add_documents(chunks, embeddings)