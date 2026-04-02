import os
from langchain.tools import tool
from typing import List
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults 


class WebSearchTool:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("TAVILY_API_KEY")
        self.web_search_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the web search tool"""
        @tool
        def search_web(query:str, max_results: int = 2) -> str:
            """Search the web for a query"""
            try:
                tavily_search = TavilySearchResults(api_key=self.api_key, max_results=max_results)
                search_result = tavily_search.invoke(query)
                return f"Following are the search results for the query '{query}': {search_result}"
            except Exception as e:
                return f"Web search cannot be performed due to {e}."
        
        return [search_web]