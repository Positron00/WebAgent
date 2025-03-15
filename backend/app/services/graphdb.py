"""
Graph Database Service
=====================

Service for storing and retrieving document relationships in a graph database.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import os
import json
from pathlib import Path
import uuid

try:
    import networkx as nx
    from py2neo import Graph, Node, Relationship
    HAS_GRAPHDB = True
except ImportError:
    HAS_GRAPHDB = False

from app.core.config import settings
from app.utils.metrics import timing_decorator

logger = logging.getLogger(__name__)

class GraphDBService:
    """Graph database service for storing document relationships."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(GraphDBService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the graph database connection."""
        self.initialized = False
        self.error = None
        self.connected = False
        self.fallback_mode = False
        
        # Check if Neo4j is available
        self.has_neo4j = HAS_GRAPHDB
        
        # Set up connection to Neo4j if URL is provided
        if settings.DATABASE_GRAPH_DB_URL:
            try:
                self.graph = Graph(settings.DATABASE_GRAPH_DB_URL)
                self.connected = True
                self.initialized = True
                logger.info(f"Connected to graph database at {settings.DATABASE_GRAPH_DB_URL}")
            except Exception as e:
                self.error = str(e)
                logger.warning(f"Failed to connect to Neo4j graph database: {str(e)}")
                logger.info("Falling back to NetworkX in-memory graph")
                self.fallback_mode = True
        else:
            logger.info("No graph database URL configured. Using NetworkX in-memory graph.")
            self.fallback_mode = True
        
        # Fall back to NetworkX if Neo4j is not available
        if self.fallback_mode:
            try:
                self.graph = nx.DiGraph()
                self.initialized = True
                logger.info("Initialized fallback NetworkX graph")
            except Exception as e:
                self.error = str(e)
                logger.error(f"Failed to initialize fallback graph: {str(e)}")
    
    @timing_decorator
    def add_document(self, document_id: str, metadata: Dict[str, Any]) -> str:
        """
        Add a document node to the graph.
        
        Args:
            document_id: The unique ID of the document
            metadata: Document metadata
            
        Returns:
            str: The node ID in the graph
        """
        if not self.initialized:
            logger.error("Graph database not initialized")
            return None
        
        try:
            if self.fallback_mode:
                # NetworkX implementation
                self.graph.add_node(document_id, 
                                    type="Document", 
                                    id=document_id,
                                    **metadata)
                return document_id
            else:
                # Neo4j implementation
                node = Node("Document", id=document_id, **metadata)
                self.graph.create(node)
                return document_id
        except Exception as e:
            logger.error(f"Error adding document to graph: {str(e)}")
            return None
    
    @timing_decorator
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a relationship between two documents.
        
        Args:
            source_id: Source document ID
            target_id: Target document ID
            relationship_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Graph database not initialized")
            return False
        
        if properties is None:
            properties = {}
            
        try:
            if self.fallback_mode:
                # NetworkX implementation
                self.graph.add_edge(source_id, target_id, 
                                   type=relationship_type,
                                   **properties)
                return True
            else:
                # Neo4j implementation
                query = (
                    f"MATCH (a), (b) "
                    f"WHERE a.id = $source_id AND b.id = $target_id "
                    f"CREATE (a)-[r:{relationship_type} $properties]->(b) "
                    f"RETURN r"
                )
                self.graph.run(query, source_id=source_id, target_id=target_id, properties=properties)
                return True
        except Exception as e:
            logger.error(f"Error adding relationship to graph: {str(e)}")
            return False
    
    @timing_decorator
    def find_related_documents(self, document_id: str, relationship_type: Optional[str] = None, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Find documents related to the given document.
        
        Args:
            document_id: Document ID to find relations for
            relationship_type: Optional specific relationship type to filter on
            depth: Maximum relationship depth
            
        Returns:
            List of related document nodes
        """
        if not self.initialized:
            logger.error("Graph database not initialized")
            return []
        
        try:
            if self.fallback_mode:
                # NetworkX implementation
                if relationship_type:
                    # Filter edges by relationship type
                    neighbors = []
                    for n in self.graph.neighbors(document_id):
                        edge_data = self.graph.get_edge_data(document_id, n)
                        if edge_data and edge_data.get("type") == relationship_type:
                            node_data = self.graph.nodes[n]
                            neighbors.append(node_data)
                    return neighbors
                else:
                    # Get all neighbors
                    return [self.graph.nodes[n] for n in self.graph.neighbors(document_id)]
            else:
                # Neo4j implementation
                rel_filter = ""
                if relationship_type:
                    rel_filter = f":{relationship_type}"
                    
                query = (
                    f"MATCH (a:Document)-[r{rel_filter}*1..{depth}]->(b:Document) "
                    f"WHERE a.id = $document_id "
                    f"RETURN b"
                )
                result = self.graph.run(query, document_id=document_id)
                return [dict(record["b"]) for record in result]
        except Exception as e:
            logger.error(f"Error finding related documents: {str(e)}")
            return []
    
    @timing_decorator
    def get_graph_status(self) -> Dict[str, Any]:
        """
        Get the status of the graph database.
        
        Returns:
            Dict with status information
        """
        status = {
            "initialized": self.initialized,
            "connected": self.connected,
            "fallback_mode": self.fallback_mode,
            "error": self.error
        }
        
        if self.initialized:
            try:
                if self.fallback_mode:
                    # NetworkX
                    status["nodes"] = self.graph.number_of_nodes()
                    status["edges"] = self.graph.number_of_edges()
                    status["type"] = "networkx"
                else:
                    # Neo4j
                    status["type"] = "neo4j"
                    status["url"] = settings.DATABASE_GRAPH_DB_URL
                    status["nodes"] = len(self.graph.nodes.match())
                    status["edges"] = len(self.graph.relationships.match())
            except Exception as e:
                status["stats_error"] = str(e)
        
        return status

# Singleton instance
_graph_db_service = None

def get_graph_db_service():
    """Get the graph database service instance."""
    global _graph_db_service
    if _graph_db_service is None:
        _graph_db_service = GraphDBService()
    return _graph_db_service 