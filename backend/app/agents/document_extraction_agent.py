"""
Document Extraction Agent
=========================

This agent specializes in extracting structured information from documents.
It can be called directly by the supervisor agent and supports multiple extraction methods:
1. NLP-based extraction with Latent Dirichlet Allocation (LDA) Bayesian Network
2. LLM-based extraction for more complex documents

The agent can route outputs directly to the Team Manager Agent for quick summarization.
"""

import logging
import time
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

# Import NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score

# Import local modules
from app.core.config import settings
from app.services.llm import get_llm
from app.agents.base_agent import BaseAgent
from app.utils.metrics import timing_decorator, log_memory_usage
from app.utils.document_utils import extract_text_from_document
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableLambda
from app.core.logger import logger, get_logger
from app.models.task import WorkflowState
from app.services.document_loader import get_document_loader_service

# Initialize logging
logger = logging.getLogger(__name__)

class DocumentExtractionAgent(BaseAgent):
    """
    Agent for document extraction with multiple processing strategies.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Document Extraction Agent with configuration settings.
        
        Args:
            config: Configuration dictionary (falls back to settings if None)
        """
        super().__init__(name="document_extraction", config=config)
        self.llm = get_llm()
        self.extraction_methods = {
            "lda": self.extract_with_lda,
            "llm": self.extract_with_llm,
            "hybrid": self.extract_with_hybrid_approach
        }
        
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Set default parameters for LDA
        self.lda_params = {
            "num_topics": config.get("lda_num_topics", 10) if config else 10,
            "passes": config.get("lda_passes", 15) if config else 15,
            "iterations": config.get("lda_iterations", 50) if config else 50,
            "chunksize": config.get("lda_chunksize", 100) if config else 100,
            "random_state": config.get("lda_random_state", 42) if config else 42
        }
        
        logger.info(f"Document Extraction Agent initialized with {len(self.extraction_methods)} methods")
    
    @timing_decorator
    @log_memory_usage
    def process_document(self, 
                         document_path: Union[str, Path], 
                         method: str = "auto",
                         extraction_params: Dict = None,
                         route_to_team_manager: bool = False) -> Dict[str, Any]:
        """
        Process a document using the specified extraction method.
        
        Args:
            document_path: Path to the document file
            method: Extraction method to use ('lda', 'llm', 'hybrid', or 'auto')
            extraction_params: Additional parameters for extraction
            route_to_team_manager: Whether to route the output to Team Manager Agent
            
        Returns:
            Extracted information and metadata
        """
        logger.info(f"Processing document: {document_path} with method: {method}")
        
        start_time = time.time()
        extraction_params = extraction_params or {}
        
        # Extract text from document
        try:
            document_text = extract_text_from_document(document_path)
            if not document_text:
                return {
                    "success": False,
                    "error": "Failed to extract text from document",
                    "processing_time": time.time() - start_time
                }
        except Exception as e:
            logger.error(f"Error extracting text from document: {str(e)}")
            return {
                "success": False,
                "error": f"Error extracting text: {str(e)}",
                "processing_time": time.time() - start_time
            }
        
        # Auto-select method based on document length and complexity
        if method == "auto":
            method = self._select_extraction_method(document_text)
            logger.info(f"Auto-selected extraction method: {method}")
        
        # Validate method
        if method not in self.extraction_methods:
            logger.error(f"Unknown extraction method: {method}")
            return {
                "success": False,
                "error": f"Unknown extraction method: {method}",
                "processing_time": time.time() - start_time
            }
        
        # Process document with selected method
        try:
            extraction_result = self.extraction_methods[method](document_text, extraction_params)
            extraction_result["method"] = method
            extraction_result["processing_time"] = time.time() - start_time
            extraction_result["success"] = True
            
            # Add metadata about the document
            extraction_result["document_metadata"] = {
                "filename": os.path.basename(document_path),
                "file_size_bytes": os.path.getsize(document_path),
                "extraction_timestamp": time.time()
            }
            
            # Route to Team Manager if requested
            if route_to_team_manager:
                logger.info("Routing extraction results to Team Manager Agent")
                self._route_to_team_manager(extraction_result)
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error during document extraction: {str(e)}")
            return {
                "success": False,
                "error": f"Extraction error: {str(e)}",
                "method": method,
                "processing_time": time.time() - start_time
            }
    
    def _select_extraction_method(self, text: str) -> str:
        """
        Automatically select the best extraction method based on document characteristics.
        
        Args:
            text: Document text
            
        Returns:
            Selected method name
        """
        # Simple heuristic based on document length and complexity
        text_length = len(text)
        sentence_count = len(sent_tokenize(text))
        
        if text_length < 5000 or sentence_count < 50:
            return "llm"  # Use LLM for shorter documents
        elif text_length > 20000 or sentence_count > 200:
            return "lda"  # Use LDA for very long documents
        else:
            return "hybrid"  # Use hybrid approach for medium-sized documents
    
    def _preprocess_text(self, text: str) -> Tuple[List[str], List[List[str]]]:
        """
        Preprocess text for NLP analysis.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentences, tokenized_sentences)
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Tokenize, remove stopwords, and lemmatize
        tokenized_sentences = []
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            # Remove stopwords and non-alphabetic tokens
            filtered_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and token.isalpha()
            ]
            tokenized_sentences.append(filtered_tokens)
        
        return sentences, tokenized_sentences
    
    @timing_decorator
    def extract_with_lda(self, text: str, params: Dict = None) -> Dict[str, Any]:
        """
        Extract information using Latent Dirichlet Allocation.
        
        Args:
            text: Document text
            params: Additional parameters for LDA
            
        Returns:
            Extracted topics and key information
        """
        logger.info("Extracting information using LDA")
        
        # Update LDA parameters with any provided overrides
        lda_params = self.lda_params.copy()
        if params:
            lda_params.update(params.get("lda_params", {}))
        
        # Preprocess text
        sentences, tokenized_sentences = self._preprocess_text(text)
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_sentences)
        corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=lda_params["num_topics"],
            passes=lda_params["passes"],
            iterations=lda_params["iterations"], 
            chunksize=lda_params["chunksize"],
            random_state=lda_params["random_state"]
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=tokenized_sentences,
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Extract topics
        topics = []
        for topic_id, topic in lda_model.print_topics():
            topic_terms = [(term.split("*")[1].strip().replace('"', ''), float(term.split("*")[0])) 
                          for term in topic.split(" + ")]
            topics.append({
                "topic_id": topic_id,
                "terms": topic_terms[:10]  # Top 10 terms
            })
        
        # Determine dominant topic for each sentence
        sentence_topics = []
        for i, sent in enumerate(sentences):
            if i < len(corpus):  # Ensure we don't go out of bounds
                bow = corpus[i]
                topic_distribution = lda_model.get_document_topics(bow)
                dominant_topic = max(topic_distribution, key=lambda x: x[1]) if topic_distribution else (0, 0)
                
                sentence_topics.append({
                    "sentence": sent,
                    "dominant_topic": dominant_topic[0],
                    "topic_probability": dominant_topic[1]
                })
        
        # Generate summary by selecting top sentences from each major topic
        important_sentences = []
        for topic_id in range(lda_params["num_topics"]):
            topic_sentences = [s for s in sentence_topics if s["dominant_topic"] == topic_id]
            sorted_sentences = sorted(topic_sentences, key=lambda x: x["topic_probability"], reverse=True)
            
            # Get top sentence for this topic if available
            if sorted_sentences:
                important_sentences.append(sorted_sentences[0]["sentence"])
        
        return {
            "topics": topics,
            "coherence_score": coherence_score,
            "important_sentences": important_sentences[:5],  # Top 5 important sentences
            "sentence_count": len(sentences),
            "dominant_topics": [t["topic_id"] for t in topics[:3]],  # Top 3 dominant topics
            "extraction_type": "lda"
        }
    
    @timing_decorator
    def extract_with_llm(self, text: str, params: Dict = None) -> Dict[str, Any]:
        """
        Extract information using the LLM.
        
        Args:
            text: Document text
            params: Additional parameters
            
        Returns:
            Structured information extracted by the LLM
        """
        logger.info("Extracting information using LLM")
        
        # Get extraction template from params or use default
        template = params.get("extraction_template") if params else None
        if not template:
            template = """
            You are an expert document analyst. Extract structured information from the following document.
            
            Your response must be ONLY a valid JSON object with these fields:
            - main_topics: List of main topics covered in the document
            - key_entities: Important entities mentioned (people, organizations, etc.)
            - summary: A concise summary (max 5 sentences)
            - key_points: List of the 5 most important points
            - document_type: The likely type of document (report, article, etc.)
            
            Document:
            {text}
            
            VERY IMPORTANT: Your response MUST be a single valid JSON object only, with no additional text, explanation, comments, or code blocks. 
            For example, your full response should look exactly like:
            {{
              "main_topics": ["Topic 1", "Topic 2"],
              "key_entities": ["Entity 1", "Entity 2"],
              "summary": "This is a summary of the document.",
              "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
              "document_type": "Report"
            }}
            
            Do not include any other text outside of this single JSON object.
            """
        
        # Truncate text if too long (based on model context limits)
        max_length = params.get("max_text_length", 6000)
        if len(text) > max_length:
            logger.warning(f"Document text too long ({len(text)} chars), truncating to {max_length}")
            text = text[:max_length] + "...[truncated]"
        
        # Prepare the prompt
        prompt = template.format(text=text)
        
        try:
            # Call the LLM using the method that works with the TogetherAI API
            # Different LLM providers may have different response formats, so handle appropriately
            response_text = self.llm.invoke(prompt)
            
            # If the response is an object with a content attribute, extract it
            if hasattr(response_text, 'content'):
                response_text = response_text.content
            
            # Try to extract JSON from the response using a more robust approach
            extracted_data = None
            
            try:
                # First try direct JSON parsing of the whole response
                try:
                    extracted_data = json.loads(response_text.strip())
                except json.JSONDecodeError:
                    # If that fails, try to find and extract the JSON portion
                    import re
                    
                    # Find any JSON-like structure using regex - use a more robust pattern
                    json_match = re.search(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})', response_text, re.DOTALL)
                    if json_match:
                        try:
                            json_str = json_match.group(1)
                            extracted_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                    
                    # If we still can't parse, try to find content between triple backticks
                    if not extracted_data:
                        code_block_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
                        if code_block_match:
                            try:
                                code_block = code_block_match.group(1).strip()
                                extracted_data = json.loads(code_block)
                            except json.JSONDecodeError:
                                pass
                
                # If all parsing attempts fail, create a structured response from the raw text
                if not extracted_data:
                    logger.warning("Failed to parse JSON from LLM response - creating structured response manually")
                    extracted_data = self._extract_structured_data_from_text(response_text)
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse JSON from LLM response: {str(parse_error)}")
                # Create a fallback structured response
                extracted_data = {
                    "main_topics": [],
                    "key_entities": [],
                    "summary": f"Failed to parse response: {str(parse_error)}",
                    "key_points": ["Error occurred during extraction"],
                    "document_type": "unknown"
                }
            
            # Handle function-call format or other inconsistent formats
            # Ensure we always return a standardized structure
            if isinstance(extracted_data, dict):
                # Check if this is a function-call format
                if "name" in extracted_data and "parameters" in extracted_data and "type" in extracted_data:
                    # Convert function-call format to structured format
                    params = extracted_data.get("parameters", {})
                    if "text" in params:
                        # This appears to be raw text, so extract structured data from it
                        logger.info("Converting function-call format to structured format")
                        structured_data = self._extract_structured_data_from_text(params.get("text", ""))
                    else:
                        # Use parameters as structured data if available
                        structured_data = params
                    
                    # Ensure required fields are present
                    for field in ["main_topics", "key_entities", "summary", "key_points", "document_type"]:
                        if field not in structured_data:
                            structured_data[field] = [] if field in ["main_topics", "key_entities", "key_points"] else ""
                            if field == "document_type":
                                structured_data[field] = "unknown"
                    
                    # Return standardized data
                    return {
                        "extraction_data": structured_data,
                        "extraction_type": "llm"
                    }
                
                # Check if we already have a properly structured response
                if any(key in extracted_data for key in ["main_topics", "key_entities", "summary", "key_points", "document_type"]):
                    # Ensure all required fields exist
                    for field in ["main_topics", "key_entities", "summary", "key_points", "document_type"]:
                        if field not in extracted_data:
                            extracted_data[field] = [] if field in ["main_topics", "key_entities", "key_points"] else ""
                            if field == "document_type":
                                extracted_data[field] = "unknown"
                    
                    return {
                        "extraction_data": extracted_data,
                        "extraction_type": "llm"
                    }
                
                # If we have other dictionary format, convert it to our standard format
                return {
                    "extraction_data": {
                        "main_topics": extracted_data.get("topics", []),
                        "key_entities": extracted_data.get("entities", []),
                        "summary": extracted_data.get("summary", ""),
                        "key_points": extracted_data.get("points", []),
                        "document_type": extracted_data.get("document_type", "unknown")
                    },
                    "extraction_type": "llm"
                }
            
            # If we got a non-dictionary response, create a structured format
            return {
                "extraction_data": {
                    "main_topics": [],
                    "key_entities": [],
                    "summary": "Failed to extract structured information from the document.",
                    "key_points": ["Unable to extract structured data"],
                    "document_type": "unknown"
                },
                "extraction_type": "llm"
            }
                
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            return {
                "extraction_data": {
                    "main_topics": [],
                    "key_entities": [],
                    "summary": f"Failed to extract information: {str(e)}",
                    "key_points": ["Error occurred during extraction"],
                    "document_type": "unknown"
                },
                "extraction_type": "llm"
            }
            
    def _extract_structured_data_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from the LLM text response when JSON parsing fails.
        
        Args:
            text: The raw text response from the LLM
            
        Returns:
            A structured dictionary with the extracted information
        """
        # Initialize default structure
        data = {
            "main_topics": [],
            "key_entities": [],
            "summary": "",
            "key_points": [],
            "document_type": "unknown"
        }
        
        try:
            import re
            
            # Try to extract main topics
            topics_match = re.search(r'"main_topics":\s*\[(.*?)\]', text, re.DOTALL)
            if topics_match:
                topics_text = topics_match.group(1)
                topics = re.findall(r'"([^"]+)"', topics_text)
                if topics:
                    data["main_topics"] = topics
            
            # Try to extract key entities
            entities_match = re.search(r'"key_entities":\s*\[(.*?)\]', text, re.DOTALL)
            if entities_match:
                entities_text = entities_match.group(1)
                entities = re.findall(r'"([^"]+)"', entities_text)
                if entities:
                    data["key_entities"] = entities
            
            # Try to extract summary
            summary_match = re.search(r'"summary":\s*"(.*?)"', text, re.DOTALL)
            if summary_match:
                data["summary"] = summary_match.group(1)
            
            # Try to extract key points
            points_match = re.search(r'"key_points":\s*\[(.*?)\]', text, re.DOTALL)
            if points_match:
                points_text = points_match.group(1)
                points = re.findall(r'"([^"]+)"', points_text)
                if points:
                    data["key_points"] = points
            
            # Try to extract document type
            doc_type_match = re.search(r'"document_type":\s*"(.*?)"', text, re.DOTALL)
            if doc_type_match:
                data["document_type"] = doc_type_match.group(1)
                
        except Exception as e:
            logger.warning(f"Error during text extraction: {str(e)}")
            
        return data
    
    @timing_decorator
    def extract_with_hybrid_approach(self, text: str, params: Dict = None) -> Dict[str, Any]:
        """
        Extract information using both LDA and LLM approaches.
        
        Args:
            text: Document text
            params: Additional parameters
            
        Returns:
            Combined extraction results
        """
        logger.info("Extracting information using hybrid approach")
        
        # First extract with LDA to get topics
        lda_result = self.extract_with_lda(text, params)
        
        # Prepare LLM prompt with LDA results
        topics_text = "\n".join([
            f"Topic {t['topic_id']}: " + ", ".join([term[0] for term in t["terms"][:5]])
            for t in lda_result["topics"][:5]
        ])
        
        important_sentences = "\n".join(lda_result["important_sentences"])
        
        hybrid_params = params.copy() if params else {}
        hybrid_params["extraction_template"] = f"""
        I've analyzed a document with topic modeling and found these main topics:
        {topics_text}
        
        These sentences appear to be important:
        {important_sentences}
        
        Based on this analysis and the full document text below, extract the most important information.
        Structure your response as JSON with these fields:
        - main_topics: List of main topics covered
        - key_entities: Important entities mentioned (people, organizations, etc.)
        - summary: A concise summary (max 5 sentences)
        - key_points: List of the 5 most important points
        - document_type: The likely type of document (report, article, etc.)
        
        Document:
        {{text}}
        
        JSON Response:
        """
        
        # Then use LLM with LDA results for enhanced extraction
        llm_result = self.extract_with_llm(text, hybrid_params)
        
        # Combine results
        combined_result = {
            "extraction_data": llm_result.get("extraction_data", {}),
            "topics": lda_result.get("topics", []),
            "coherence_score": lda_result.get("coherence_score", 0),
            "important_sentences": lda_result.get("important_sentences", []),
            "llm_usage": llm_result.get("llm_usage", {}),
            "extraction_type": "hybrid"
        }
        
        return combined_result
    
    def _route_to_team_manager(self, extraction_result: Dict[str, Any]) -> None:
        """
        Route extraction results to the Team Manager Agent.
        
        Args:
            extraction_result: Results from document extraction
        """
        # This would call the Team Manager Agent's API
        # For now, just log that this would happen
        logger.info("Would route results to Team Manager Agent")
        
        # In a real implementation, this might be:
        # response = requests.post(
        #     f"{settings.team_manager_url}/summarize",
        #     json={"extraction_result": extraction_result}
        # )
        
        # Return handling would be implemented as needed 

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the document extraction agent on the given workflow state.
        
        Args:
            state: Current workflow state with query and document_id
            
        Returns:
            Updated workflow state with extraction results
        """
        document_id = state.inputs.get("document_id")
        if not document_id:
            state.error = "Document ID is required for extraction"
            return state
            
        # Get the document content from the vector database
        doc_service = get_document_loader_service()
        
        # Since we don't have a direct get_document method, we'll use the vector database
        # to retrieve the document content
        from app.services.vectordb import get_vectordb_service
        vectordb = get_vectordb_service()
        
        # Retrieve document from vector database
        try:
            document = vectordb.get_document_by_artifact_id(document_id)
            if not document:
                state.error = f"Document with ID {document_id} not found in vector database"
                return state
                
            # Extract text content from document
            text = document.page_content
            
            # Select extraction method
            method = state.inputs.get("extraction_method", "auto")
            if method == "auto":
                method = self._select_extraction_method(text)
            
            # Extract information using selected method
            extraction_params = state.inputs.get("extraction_params", {})
            
            if method not in self.extraction_methods:
                state.error = f"Unknown extraction method: {method}"
                return state
                
            # Execute extraction
            extraction_result = self.extraction_methods[method](text, extraction_params)
            
            # Update state with results
            state.output = extraction_result
            
            # Route to team manager if requested
            if state.inputs.get("route_to_team_manager", False):
                self._route_to_team_manager(extraction_result)
                
            return state
        except Exception as e:
            state.error = f"Error retrieving document: {str(e)}"
            return state

def get_document_extraction_agent() -> Any:
    """
    Factory function to get a configured DocumentExtractionAgent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    agent = DocumentExtractionAgent()
    
    # Return a callable function instead of the agent directly
    async def document_extraction_function(state: WorkflowState) -> WorkflowState:
        """Callable wrapper for the document extraction agent"""
        return await agent.run(state)
    
    return document_extraction_function 