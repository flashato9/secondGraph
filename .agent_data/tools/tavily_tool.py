
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

class TavilySearchTool:
    def __init__(self):
        # Fetch the API key from the environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in .env file.")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query):
        """
        Performs a search using the Tavily API and returns the results.
        """
        try:
            response = self.client.search(query=query)
            return response
        except Exception as e:
            return f"Error performing search: {e}"
