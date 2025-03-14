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
from sklearn.decomposition import LatentDirichletAllocation

# Import local modules
from app.core.config import settings
from app.services.llm import get_llm
from app.agents.base_agent import BaseAgent
from app.utils.metrics import timing_decorator, log_memory_usage
from app.utils.document_utils import extract_text_from_document
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableLambda

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
            "num_topics": config.get("lda_num_topics", 10),
            "passes": config.get("lda_passes", 15),
            "iterations": config.get("lda_iterations", 50),
            "chunksize": config.get("lda_chunksize", 100),
            "random_state": config.get("lda_random_state", 42)
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
            Extract the most important information from the following document.
            Structure your response as JSON with these fields:
            - main_topics: List of main topics covered
            - key_entities: Important entities mentioned (people, organizations, etc.)
            - summary: A concise summary (max 5 sentences)
            - key_points: List of the 5 most important points
            - document_type: The likely type of document (report, article, etc.)
            
            Document:
            {text}
            
            JSON Response:
            """
        
        # Truncate text if too long (based on model context limits)
        max_length = params.get("max_text_length", 6000)
        if len(text) > max_length:
            logger.warning(f"Document text too long ({len(text)} chars), truncating to {max_length}")
            text = text[:max_length] + "...[truncated]"
        
        # Call the LLM
        prompt = template.format(text=text)
        
        llm_response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Extract structured information from documents accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=params.get("temperature", 0.1),
            top_p=params.get("top_p", 0.95),
            max_tokens=params.get("max_tokens", 1500)
        )
        
        response_text = llm_response.choices[0].message.content
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = response_text[json_start:json_end+1]
                extraction_data = json.loads(json_str)
            else:
                # If no JSON found, structure the response manually
                extraction_data = {
                    "main_topics": [],
                    "key_entities": [],
                    "summary": response_text[:500],  # Use first 500 chars as summary
                    "key_points": [],
                    "document_type": "unknown"
                }
                
            extraction_data["llm_usage"] = {
                "prompt_tokens": llm_response.usage.prompt_tokens,
                "completion_tokens": llm_response.usage.completion_tokens,
                "total_tokens": llm_response.usage.total_tokens
            }
            
            return {
                "extraction_data": extraction_data,
                "extraction_type": "llm"
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            # Return raw response as fallback
            return {
                "extraction_data": {
                    "raw_response": response_text,
                    "error": "Failed to parse structured data"
                },
                "llm_usage": {
                    "prompt_tokens": llm_response.usage.prompt_tokens,
                    "completion_tokens": llm_response.usage.completion_tokens,
                    "total_tokens": llm_response.usage.total_tokens
                },
                "extraction_type": "llm"
            }
    
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