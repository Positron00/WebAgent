"""
Web search service for the WebAgent backend.
Provides access to Tavily search API.
"""
from typing import Dict, List, Optional, Any
import logging
import json

from tavily import TavilyClient

from app.core.config import settings

logger = logging.getLogger(__name__)

class WebSearchService:
    """
    Web search service using the Tavily API.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(WebSearchService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the web search client."""
        self.initialized = False
        
        if not settings.TAVILY_API_KEY:
            logger.warning("Tavily API key not set. Web search will not be available.")
            return
        
        try:
            self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            self.initialized = True
            logger.info("Tavily web search client initialized.")
        except Exception as e:
            logger.error(f"Error initializing Tavily client: {str(e)}")
    
    async def search(
        self, 
        query: str, 
        search_depth: str = "basic", 
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search using Tavily.
        
        Args:
            query: The search query
            search_depth: Search depth ("basic" or "comprehensive")
            max_results: Maximum number of results to return
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            
        Returns:
            A list of search results
        """
        if not self.initialized:
            logger.error("Tavily client not initialized. Cannot perform search.")
            return []
        
        try:
            # Perform the search
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results
            }
            
            if include_domains:
                search_params["include_domains"] = include_domains
                
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            logger.info(f"Performing Tavily search: {query}")
            response = self.client.search(**search_params)
            
            # Extract and format the results
            results = response.get("results", [])
            
            logger.info(f"Tavily search completed. Found {len(results)} results.")
            return results
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            return []

def get_web_search_service() -> WebSearchService:
    """
    Get the web search service instance.
    
    Returns:
        A WebSearchService instance
    """
    return WebSearchService()

async def perform_web_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform a web search using Tavily.
    
    Args:
        query: The search query
        search_depth: Search depth ("basic" or "comprehensive")
        max_results: Maximum number of results to return
        
    Returns:
        A list of search results
    """
    service = get_web_search_service()
    return await service.search(query, search_depth, max_results)

async def get_web_search_status() -> Dict[str, Any]:
    """
    Check the status of the web search service.
    
    Returns:
        Dict with status information
    """
    if not settings.TAVILY_API_KEY:
        return {"status": "unavailable", "error": "API key not configured"}
    
    try:
        service = get_web_search_service()
        if not service.initialized:
            return {"status": "error", "error": "Failed to initialize Tavily client"}
        
        # Try a simple search to verify functionality
        results = await service.search("test", max_results=1)
        
        if results:
            return {
                "status": "ok",
                "provider": "tavily"
            }
        else:
            return {
                "status": "error",
                "error": "No search results returned",
                "provider": "tavily"
            }
    except Exception as e:
        logger.error(f"Web search health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "provider": "tavily"
        } 