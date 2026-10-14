"""
Web search service for the WebAgent backend.
Provides access to Tavily search API.
"""
from typing import Dict, List, Optional, Any
import asyncio
import logging
import json
import re

from tavily import TavilyClient

from app.core.config import settings
from app.core.log_context import describe_text

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """A search that failed or timed out (IMP-055), as opposed to one that found nothing."""


def _log_text(action: str, text: str) -> None:
    """Log a search query's length; the text itself only at DEBUG in dev (IMP-070)."""
    logger.info(f"{action} ({describe_text(text)})")
    if settings.WEBAGENT_ENV == "dev":
        logger.debug(f"{action}: {text}")

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
        """Initialize the Tavily client."""
        self.initialized = False
        self.tavily_client = None
        if not settings.TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY not found in environment")
            return

        try:
            self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            self.initialized = True
            logger.info("Tavily web search client initialized.")
        except Exception as e:
            logger.error(f"Error initializing Tavily client: {str(e)}")
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize the query to prevent bad requests."""
        if not query:
            return ""
        
        # Remove special characters that might cause problems
        sanitized = re.sub(r'[^\w\s\-\,\.\?\!]', ' ', query)
        
        # Limit query length to avoid exceeding limits
        max_length = 300
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        # Ensure proper spacing
        sanitized = ' '.join(sanitized.split())
        
        return sanitized

    async def _tavily_search(self, **kwargs) -> Dict:
        """One blocking Tavily call in a worker thread, limited to SEARCH_TIMEOUT_SECONDS.

        On timeout the thread keeps running until Tavily answers; only the wait is cut short.
        """
        return await asyncio.wait_for(
            asyncio.to_thread(self.tavily_client.search, **kwargs),
            timeout=settings.SEARCH_TIMEOUT_SECONDS,
        )

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Perform a web search.
        
        Args:
            query: The search query.
            search_depth: "basic" or "comprehensive".
            max_results: Maximum number of results to return (1-15).
            include_domains: List of domains to include in search.
            exclude_domains: List of domains to exclude from search.
            
        Returns:
            List of search results, each containing title, url, and content.
        """
        if not self.initialized or not self.tavily_client:
            logger.warning("Tavily client not initialized")
            raise SearchError("web search is not configured")

        # Sanitize query to prevent bad requests
        sanitized_query = self._sanitize_query(query)
        if not sanitized_query:
            logger.warning(f"Empty query after sanitization ({describe_text(query)}); using default")
            sanitized_query = "general information"
            
        # Ensure search_depth is valid
        valid_depths = ["basic", "advanced"]
        if search_depth not in valid_depths:
            logger.warning(f"Invalid search_depth: {search_depth}, defaulting to 'basic'")
            search_depth = "basic"
            
        # Ensure max_results is within valid range
        if max_results < 1:
            max_results = 1
        elif max_results > 15:
            max_results = 15

        _log_text("Searching", sanitized_query)
        try:
            response = await self._tavily_search(
                query=sanitized_query,
                search_depth=search_depth,
                max_results=max_results,
                include_domains=include_domains if include_domains else None,
                exclude_domains=exclude_domains if exclude_domains else None,
            )
            results = response.get("results", [])
            logger.info(f"Found {len(results)} results")
            return results
        except asyncio.TimeoutError:
            logger.warning(f"Search timed out after {settings.SEARCH_TIMEOUT_SECONDS} s")
            raise SearchError(f"timed out after {settings.SEARCH_TIMEOUT_SECONDS} s")
        except Exception as e:
            error_details = str(e)
            logger.error(f"Error during Tavily search: {error_details}")

            # Log detailed error information for debugging
            if hasattr(e, 'response') and getattr(e, 'response', None) is not None:
                response = getattr(e, 'response')
                if hasattr(response, 'text'):
                    # The body can echo the query, which only dev may log (IMP-070).
                    if settings.WEBAGENT_ENV == "dev":
                        logger.debug(f"Response text: {response.text}")
                    else:
                        logger.error(f"Response text ({describe_text(response.text)})")
                if hasattr(response, 'status_code'):
                    logger.error(f"Status code: {response.status_code}")

            # Try one more time with a simplified query if the error was a 400
            simplified_query = " ".join(sanitized_query.split()[:5])
            if "400" not in error_details or simplified_query == sanitized_query:
                raise SearchError(error_details[:300]) from e
            _log_text("Retrying with a simplified query", simplified_query)
            try:
                response = await self._tavily_search(
                    query=simplified_query,
                    search_depth="basic",  # Always use basic for retry
                    max_results=max(1, max_results - 2)  # Reduce result count
                )
            except asyncio.TimeoutError:
                logger.warning(f"Search retry timed out after {settings.SEARCH_TIMEOUT_SECONDS} s")
                raise SearchError(f"retry timed out after {settings.SEARCH_TIMEOUT_SECONDS} s")
            except Exception as retry_error:
                logger.error(f"Retry also failed: {str(retry_error)}")
                raise SearchError(str(retry_error)[:300]) from retry_error
            results = response.get("results", [])
            logger.info(f"Retry found {len(results)} results")
            return results

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

async def is_web_search_available() -> bool:
    """Check if web search is available.
    
    Returns:
        True if web search is available, False otherwise.
    """
    service = get_web_search_service()
    if not service.initialized or not service.tavily_client:
        return False
    
    # Try a simple search to verify
    try:
        results = await service.search("test", max_results=1)
        return len(results) > 0
    except Exception as e:
        logger.error(f"Web search test failed: {str(e)}")
        return False 