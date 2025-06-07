import os
from typing import List
from pathlib import Path
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import sys
import time
import re
import random
from prompts import SECURITY_AUDIT_PROMPT

# Load environment variables
load_dotenv()

class ReportRAG:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "security-reports"
        
        # Initialize Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=groq_key)
        
        # Initialize or get index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or get the Pinecone index."""
        try:
            # Check if index exists
            if not self.pc.has_index(self.index_name):
                print("Creating new Pinecone index...", file=sys.stderr)
                self.pc.create_index_for_model(
                    name=self.index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "llama-text-embed-v2",
                        "field_map": {"text": "chunk_text"}
                    }
                )
            self.index = self.pc.Index(self.index_name)
            print("Connected to Pinecone index successfully", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing Pinecone index: {str(e)}", file=sys.stderr)
            raise

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def split_into_chunks(self, text: str, max_chunk_size: int = 40000) -> List[str]:
        """Split text into chunks of maximum size, trying to break at paragraph boundaries."""
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                if current_chunk:  # If we have content, save it as a chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If a single paragraph is too long, split it by sentences
                if len(paragraph) > max_chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 > max_chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            temp_chunk += " " + sentence if temp_chunk else sentence
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def load_markdown_files(self) -> List[dict]:
        """Load all markdown files from the reports directory recursively."""
        if not os.path.exists(self.reports_dir):
            print(f"Warning: Reports directory {self.reports_dir} does not exist", file=sys.stderr)
            return []
            
        documents = []
        for root, _, files in os.walk(self.reports_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Skip README.md and CONTRIBUTING.md
                            if file.lower() in ['readme.md', 'contributing.md']:
                                continue
                                
                            # Get relative path for cleaner metadata
                            rel_path = str(file_path.relative_to(self.reports_dir))
                            
                            # Split content into chunks if needed
                            chunks = self.split_into_chunks(content)
                            
                            # Create a document for each chunk
                            for i, chunk in enumerate(chunks):
                                documents.append({
                                    "_id": f"doc_{len(documents)}",
                                    "chunk_text": chunk,
                                    "source": f"{rel_path}#chunk_{i+1}"
                                })
                                
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}", file=sys.stderr)
        return documents

    def upsert_with_retry(self, batch: List[dict], max_retries: int = 3) -> bool:
        """Upsert a batch of documents with retry logic."""
        for attempt in range(max_retries):
            try:
                # Reduce batch size if we're having connection issues
                if attempt > 0:
                    # Split the batch in half for retry
                    mid = len(batch) // 2
                    first_half = batch[:mid]
                    second_half = batch[mid:]
                    
                    # Try first half
                    print(f"Retry attempt {attempt + 1}: Trying with reduced batch size of {len(first_half)}", file=sys.stderr)
                    self.index.upsert_records("security-reports", first_half)
                    
                    # Add a small delay between sub-batches
                    time.sleep(2)
                    
                    # Try second half
                    print(f"Retry attempt {attempt + 1}: Trying with reduced batch size of {len(second_half)}", file=sys.stderr)
                    self.index.upsert_records("security-reports", second_half)
                else:
                    self.index.upsert_records("security-reports", batch)
                return True
            except Exception as e:
                error_msg = str(e)
                if any(err in error_msg for err in ["RESOURCE_EXHAUSTED", "Too Many Requests", "Broken pipe", "Connection"]):
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff and jitter
                        delay = self.batch_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Connection issue or rate limit hit, waiting {delay:.1f} seconds before retry...", file=sys.stderr)
                        time.sleep(delay)
                        continue
                print(f"Error during upsert: {error_msg}", file=sys.stderr)
                raise
        return False

    def initialize_rag(self):
        """Initialize the RAG system with the markdown files."""
        try:
            # Check if index is empty
            stats = self.index.describe_index_stats()
            if stats.total_vector_count > 0:
                print("Using existing Pinecone index...", file=sys.stderr)
                return
                
            print("Indexing documents in Pinecone...", file=sys.stderr)
            # Load and process documents
            documents = self.load_markdown_files()
            if not documents:
                print("Warning: No markdown files found to process", file=sys.stderr)
                return
                
            print(f"Processing {len(documents)} chunks from markdown files...", file=sys.stderr)
            
            # Calculate tokens per batch and adjust batch size
            total_tokens = sum(self.estimate_tokens(doc["chunk_text"]) for doc in documents)
            avg_tokens_per_doc = total_tokens / len(documents)
            self.tokens_per_batch = min(96 * avg_tokens_per_doc, self.max_tokens_per_minute)
            batch_size = min(48, int(self.tokens_per_batch / avg_tokens_per_doc))  # Reduced from 96 to 48
            
            print(f"Using batch size of {batch_size} documents", file=sys.stderr)
                
            # Upsert records in batches with rate limiting
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                if self.upsert_with_retry(batch):
                    print(f"Upserted batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}", file=sys.stderr)
                else:
                    print(f"Failed to upsert batch {i//batch_size + 1} after retries", file=sys.stderr)
                
                # Add delay between batches to respect rate limits
                if i + batch_size < len(documents):
                    time.sleep(self.batch_delay)
            
            # Wait for vectors to be indexed
            print("Waiting for vectors to be indexed...", file=sys.stderr)
            time.sleep(10)
            
            # Check stats
            stats = self.index.describe_index_stats()
            print(f"Successfully indexed {stats.total_vector_count} chunks", file=sys.stderr)
                
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}", file=sys.stderr)
            raise

    def chunk_text(self, text: str, max_tokens: int = 800) -> List[str]:
        """Split text into chunks that fit within token limits."""
        if self.estimate_tokens(text) <= max_tokens:
            return [text]
            
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if self.estimate_tokens(current_chunk + sentence) <= max_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system with a question and get a summarized response."""
        try:
            # Search the index
            results = self.index.search(
                namespace="security-reports",
                query={
                    "top_k": top_k,
                    "inputs": {
                        'text': question
                    }
                }
            )
            
            if not results['result']['hits']:
                return "No relevant documents found to answer your question."
            
            # Prepare context from retrieved documents
            all_chunks = []
            for hit in results['result']['hits']:
                source = hit['fields'].get('source', 'Unknown')
                content = hit['fields'].get('chunk_text', '')
                # Split content into smaller chunks if needed
                content_chunks = self.chunk_text(content)
                for chunk in content_chunks:
                    all_chunks.append(f"Source: {source}\nContent: {chunk}")
            
            # Combine chunks into context, ensuring we don't exceed token limits
            context = "\n\n".join(all_chunks)
            
            # Format the prompt using the security audit template
            prompt = SECURITY_AUDIT_PROMPT.format(
                context=context,
                question=question
            )

            # Get response from Groq
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are a senior Web3 security auditor specializing in smart contract security, DeFi protocols, and blockchain applications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            if not completion.choices or not completion.choices[0].message:
                return "No response generated from the model."
                
            return completion.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg, file=sys.stderr)
            return error_msg 