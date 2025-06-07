import os
from typing import List
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
from prompts import SECURITY_AUDIT_PROMPT
import json
import sys
import tiktoken

# Load environment variables
load_dotenv()

class ReportRAG:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None
        # Initialize tokenizer for counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # Set token limits
        self.max_context_tokens = 20000  # Leave room for prompt and response
        self.max_response_tokens = 1024

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def truncate_context(self, documents: List[str], max_tokens: int) -> str:
        """Truncate context to fit within token limit."""
        context = ""
        current_tokens = 0
        
        for doc in documents:
            doc_tokens = self.count_tokens(doc)
            if current_tokens + doc_tokens > max_tokens:
                break
            context += doc + "\n"
            current_tokens += doc_tokens
            
        return context.strip()

    def load_markdown_files(self) -> List[str]:
        """Load all markdown files from the reports directory recursively."""
        if not os.path.exists(self.reports_dir):
            print(f"Warning: Reports directory {self.reports_dir} does not exist", file=sys.stderr)
            return []
            
        markdown_files = []
        for root, _, files in os.walk(self.reports_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            markdown_files.append(f.read())
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}", file=sys.stderr)
        return markdown_files

    def initialize_rag(self):
        """Initialize the RAG system with the markdown files."""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(
                name="security_reports",
                embedding_function=self.embeddings
            )
            print("Loaded existing collection...", file=sys.stderr)
        except Exception as e:
            print(f"Creating new collection: {str(e)}", file=sys.stderr)
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="security_reports",
                embedding_function=self.embeddings
            )
            
            # Load and process documents
            documents = self.load_markdown_files()
            if not documents:
                print("Warning: No markdown files found to process", file=sys.stderr)
                return
                
            for i, doc in enumerate(documents):
                try:
                    self.collection.add(
                        documents=[doc],
                        ids=[f"doc_{i}"]
                    )
                except Exception as e:
                    print(f"Error adding document {i}: {str(e)}", file=sys.stderr)

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self.collection:
            raise ValueError("RAG system not initialized. Call initialize_rag() first.")
        
        try:
            # Search for relevant documents
            results = self.collection.query(
                query_texts=[question],
                n_results=5  # Reduced from 15 to 5
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant documents found to answer your question."
            
            # Prepare and truncate context
            context = self.truncate_context(results["documents"][0], self.max_context_tokens)
            
            # Format the prompt with context and question
            prompt = SECURITY_AUDIT_PROMPT.format(
                context=context,
                question=question
            )
            
            # Get response from Groq
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are a security audit expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=self.max_response_tokens
            )
            
            if not completion.choices or not completion.choices[0].message:
                return "No response generated from the model."
                
            return completion.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg, file=sys.stderr)
            return error_msg 