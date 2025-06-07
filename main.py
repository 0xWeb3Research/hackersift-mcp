from mcp.server.fastmcp import FastMCP
import math
from rag_system import ReportRAG
import json
import sys

mcp = FastMCP("Report RAG System")

# Initialize RAG system
rag = ReportRAG()
rag.initialize_rag()

@mcp.tool()
def query_reports(question: str) -> str:
    """Query the reports database with a question about security audits and findings"""
    try:
        response = rag.query(question)
        # Ensure the response is a string and properly formatted
        if isinstance(response, str):
            return response
        return json.dumps(response)
    except Exception as e:
        error_msg = f"Error in query_reports: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

def test_query():
    """Test function to directly query the RAG system"""
    print("Testing RAG system...")
    test_question = "What security issues in block.timestamp?"
    print(f"Question: {test_question}")
    response = query_reports(test_question)
    print(f"Response: {response}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_query()
    else:
        try:
            print("Starting MCP server...", file=sys.stderr)
            mcp.run(transport="stdio")
        except Exception as e:
            print(f"Error in main: {str(e)}", file=sys.stderr)
            sys.exit(1)
