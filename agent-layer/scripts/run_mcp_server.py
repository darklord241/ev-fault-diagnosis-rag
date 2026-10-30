import sys
import os

# Note
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_server.server import run_server

if __name__ == "__main__":
    run_server()
