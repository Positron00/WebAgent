"""
LDA Document Extractor
======================

A specialized service for extracting document information using 
Latent Dirichlet Allocation (LDA) with Bayesian Network techniques.

This service provides advanced topic modeling capabilities for document analysis,
optimized for performance and accuracy.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# Import ML & NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

# Import local modules
from backend.app.utils.document_utils import split_document
from backend.app.core.config import settings

# Initialize logging
logger = logging.getLogger(__name__)

class LDADocumentExtractor:
    """
    Document information extractor using Latent Dirichlet Allocation (LDA) with Bayesian Network techniques.
    
    This class implements advanced topic modeling to extract structured information from documents,
    with a focus on identifying key topics, entities, and relationships.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the LDA Document Extractor.
        
        Args:
            config: Configuration dictionary with LDA parameters
        """
        self.config = config or {}
        
        # Set default LDA parameters
        self.lda_params = {
            "num_topics": self.config.get("lda_num_topics", 10),
            "passes": self.config.get("lda_passes", 15),
            "iterations": self.config.get("lda_iterations", 50),
            "chunksize": self.config.get("lda_chunksize", 100),
            "decay": self.config.get("lda_decay", 0.5),
            "offset": self.config.get("lda_offset", 1.0),
            "alpha": self.config.get("lda_alpha", "auto"),
            "eta": self.config.get("lda_eta", "auto"),
            "eval_every": self.config.get("lda_eval_every", None),
            "random_state": self.config.get("lda_random_state", 42),
            "workers": self.config.get("lda_workers", 3),
            "minimum_probability": self.config.get("lda_minimum_probability", 0.01)
        }
        
        # Set up preprocessing options
        self.preprocessing = {
            "remove_stopwords": self.config.get("remove_stopwords", True),
            "lemmatize": self.config.get("lemmatize", True),
            "min_token_length": self.config.get("min_token_length", 3),
            "max_token_length": self.config.get("max_token_length", 50),
            "min_df": self.config.get("min_df", 2),
            "max_df": self.config.get("max_df", 0.95)
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
        
        # Initialize utilities
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Optional: Add domain-specific stopwords
        if self.config.get("domain_stopwords"):
            self.stop_words.update(self.config.get("domain_stopwords"))
        
        logger.info(f"LDA Document Extractor initialized with {self.lda_params['num_topics']} topics")
    
    def extract_information(self, text: str, extract_params: Dict = None) -> Dict[str, Any]:
        """
        Extract structured information from document text using LDA.
        
        Args:
            text: Document text
            extract_params: Additional parameters for extraction
            
        Returns:
            Dictionary with extracted information and metadata
        """
        extract_params = extract_params or {}
        start_time = time.time()
        
        # Update LDA parameters if provided
        lda_params = self.lda_params.copy()
        if "lda_params" in extract_params:
            lda_params.update(extract_params["lda_params"])
        
        # Process document
        try:
            # Preprocess text and create corpus
            sentences, tokenized_sentences = self._preprocess_text(text)
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(tokenized_sentences)
            corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_sentences]
            
            # Train LDA model
            lda_model = self._train_lda_model(corpus, dictionary, lda_params)
            
            # Extract topics
            topics = self._extract_topics(lda_model, dictionary)
            
            # Calculate coherence score
            coherence = self._calculate_coherence(lda_model, tokenized_sentences, dictionary)
            
            # Extract key sentences
            sentence_topics = self._extract_sentence_topics(sentences, corpus, lda_model)
            key_sentences = self._select_key_sentences(sentence_topics, lda_params["num_topics"])
            
            # Extract entities using Bayesian approach
            key_entities = self._extract_entities(text, lda_model, dictionary, tokenized_sentences)
            
            # Generate summary
            summary = self._generate_summary(sentence_topics, lda_model, lda_params["num_topics"])
            
            # Construct result
            result = {
                "success": True,
                "processing_time": time.time() - start_time,
                "topics": topics,
                "coherence_score": coherence,
                "top_sentences": key_sentences,
                "key_entities": key_entities,
                "summary": summary,
                "document_stats": {
                    "sentence_count": len(sentences),
                    "word_count": sum(len(tokens) for tokens in tokenized_sentences),
                    "vocabulary_size": len(dictionary)
                },
                "bayesian_network": {
                    "topic_word_matrix": self._get_topic_word_matrix(lda_model, dictionary, top_n=20),
                    "topic_correlations": self._calculate_topic_correlations(lda_model, corpus)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during LDA extraction: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"LDA extraction error: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _preprocess_text(self, text: str) -> Tuple[List[str], List[List[str]]]:
        """
        Preprocess text for LDA analysis.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (original_sentences, tokenized_sentences)
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Tokenize and clean
        tokenized_sentences = []
        for sentence in sentences:
            # Tokenize
            tokens = word_tokenize(sentence.lower())
            
            # Apply preprocessing based on configuration
            filtered_tokens = []
            for token in tokens:
                # Skip non-alphabetic tokens and those not meeting length criteria
                if not token.isalpha():
                    continue
                
                if len(token) < self.preprocessing["min_token_length"] or len(token) > self.preprocessing["max_token_length"]:
                    continue
                
                # Remove stopwords if configured
                if self.preprocessing["remove_stopwords"] and token in self.stop_words:
                    continue
                
                # Lemmatize if configured
                if self.preprocessing["lemmatize"]:
                    token = self.lemmatizer.lemmatize(token)
                
                filtered_tokens.append(token)
            
            tokenized_sentences.append(filtered_tokens)
        
        return sentences, tokenized_sentences
    
    def _train_lda_model(self, corpus: List, dictionary: corpora.Dictionary, lda_params: Dict) -> LdaModel:
        """
        Train the LDA model.
        
        Args:
            corpus: Document corpus
            dictionary: Word dictionary
            lda_params: LDA parameters
            
        Returns:
            Trained LDA model
        """
        logger.info(f"Training LDA model with {lda_params['num_topics']} topics")
        
        # Determine if we should use multicore
        use_multicore = lda_params.get("workers", 1) > 1
        
        if use_multicore:
            model = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=lda_params["num_topics"],
                passes=lda_params["passes"],
                iterations=lda_params["iterations"],
                chunksize=lda_params["chunksize"],
                alpha=lda_params["alpha"],
                eta=lda_params["eta"],
                eval_every=lda_params["eval_every"],
                workers=lda_params["workers"],
                random_state=lda_params["random_state"],
                minimum_probability=lda_params["minimum_probability"]
            )
        else:
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=lda_params["num_topics"],
                passes=lda_params["passes"],
                iterations=lda_params["iterations"],
                chunksize=lda_params["chunksize"],
                alpha=lda_params["alpha"],
                eta=lda_params["eta"],
                eval_every=lda_params["eval_every"],
                random_state=lda_params["random_state"],
                minimum_probability=lda_params["minimum_probability"]
            )
        
        return model
    
    def _extract_topics(self, lda_model: LdaModel, dictionary: corpora.Dictionary) -> List[Dict]:
        """
        Extract topics from the LDA model.
        
        Args:
            lda_model: Trained LDA model
            dictionary: Word dictionary
            
        Returns:
            List of topics with terms and weights
        """
        topics = []
        
        for topic_id in range(lda_model.num_topics):
            topic_terms = lda_model.get_topic_terms(topic_id, topn=20)  # Get top 20 terms
            
            # Format topic terms
            formatted_terms = []
            for term_id, weight in topic_terms:
                term = dictionary[term_id]
                formatted_terms.append({"term": term, "weight": float(weight)})
            
            topics.append({
                "topic_id": topic_id,
                "terms": formatted_terms
            })
        
        return topics
    
    def _calculate_coherence(self, lda_model: LdaModel, 
                            tokenized_sentences: List[List[str]], 
                            dictionary: corpora.Dictionary) -> float:
        """
        Calculate the coherence score for the LDA model.
        
        Args:
            lda_model: Trained LDA model
            tokenized_sentences: Preprocessed sentences
            dictionary: Word dictionary
            
        Returns:
            Coherence score
        """
        try:
            coherence_model = CoherenceModel(
                model=lda_model, 
                texts=tokenized_sentences,
                dictionary=dictionary, 
                coherence='c_v'
            )
            return float(coherence_model.get_coherence())
        except Exception as e:
            logger.warning(f"Error calculating coherence: {str(e)}")
            return 0.0
    
    def _extract_sentence_topics(self, sentences: List[str], 
                                corpus: List, 
                                lda_model: LdaModel) -> List[Dict]:
        """
        Extract the dominant topic for each sentence.
        
        Args:
            sentences: Original sentences
            corpus: Document corpus
            lda_model: Trained LDA model
            
        Returns:
            List of sentences with their dominant topics
        """
        sentence_topics = []
        
        for i, (sent, bow) in enumerate(zip(sentences, corpus)):
            # Get topic distribution for this sentence
            topic_distribution = lda_model.get_document_topics(bow)
            
            # Find dominant topic
            if not topic_distribution:
                dominant_topic = None
                topic_probability = 0.0
            else:
                dominant_topic = max(topic_distribution, key=lambda x: x[1])
                topic_probability = dominant_topic[1]
                dominant_topic = dominant_topic[0]
            
            sentence_topics.append({
                "sentence_id": i,
                "sentence": sent,
                "dominant_topic": dominant_topic,
                "topic_probability": float(topic_probability),
                "length": len(sent)
            })
        
        return sentence_topics
    
    def _select_key_sentences(self, sentence_topics: List[Dict], num_topics: int) -> List[Dict]:
        """
        Select key sentences from each topic.
        
        Args:
            sentence_topics: Sentences with topic information
            num_topics: Number of topics
            
        Returns:
            List of key sentences for each topic
        """
        key_sentences = []
        
        # Group sentences by topic
        for topic_id in range(num_topics):
            topic_sentences = [s for s in sentence_topics if s["dominant_topic"] == topic_id]
            
            # Sort by topic probability (highest first)
            sorted_sentences = sorted(topic_sentences, key=lambda x: x["topic_probability"], reverse=True)
            
            # Get top 2 sentences for this topic if available
            for i, sent in enumerate(sorted_sentences[:2]):
                if sent["topic_probability"] > 0.3:  # Only include if probability is significant
                    key_sentences.append({
                        "topic_id": topic_id,
                        "sentence": sent["sentence"],
                        "probability": sent["topic_probability"],
                        "rank_in_topic": i + 1
                    })
        
        # Sort key sentences by topic_id and rank
        key_sentences.sort(key=lambda x: (x["topic_id"], x["rank_in_topic"]))
        
        return key_sentences
    
    def _extract_entities(self, text: str, 
                         lda_model: LdaModel, 
                         dictionary: corpora.Dictionary,
                         tokenized_sentences: List[List[str]]) -> List[Dict]:
        """
        Extract key entities using Bayesian approach.
        
        Args:
            text: Document text
            lda_model: Trained LDA model
            dictionary: Word dictionary
            tokenized_sentences: Preprocessed sentences
            
        Returns:
            List of extracted entities with metadata
        """
        try:
            # Use NLTK's Named Entity Recognition if available
            entities = []
            
            # Try to load the NER module and apply it
            try:
                from nltk import ne_chunk
                nltk.download('maxent_ne_chunker')
                nltk.download('words')
                
                # Perform simple NER
                chunks = ne_chunk(nltk.pos_tag(word_tokenize(text)))
                
                # Extract named entities
                named_entities = {}
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_name = ' '.join([c[0] for c in chunk])
                        entity_type = chunk.label()
                        
                        if entity_name not in named_entities:
                            named_entities[entity_name] = {
                                "name": entity_name,
                                "type": entity_type,
                                "count": 1
                            }
                        else:
                            named_entities[entity_name]["count"] += 1
                
                entities = list(named_entities.values())
                
            except (ImportError, LookupError):
                # Fallback to frequency-based extraction
                logger.warning("NER not available, using frequency-based entity extraction")
                
                # Use topic terms as proxy for entities
                entity_candidates = {}
                
                # Go through top terms in each topic
                for topic_id in range(lda_model.num_topics):
                    topic_terms = lda_model.get_topic_terms(topic_id, topn=10)
                    
                    # Use terms as entities
                    for term_id, weight in topic_terms:
                        term = dictionary[term_id]
                        
                        # Skip very short terms
                        if len(term) < 4:
                            continue
                        
                        # Count term frequency across document
                        count = sum(1 for tokens in tokenized_sentences if term in tokens)
                        
                        if term not in entity_candidates:
                            entity_candidates[term] = {
                                "name": term,
                                "type": "TOPIC_TERM",
                                "count": count,
                                "weight": float(weight),
                                "topic_id": topic_id
                            }
                        else:
                            # Update if this topic has higher weight
                            if weight > entity_candidates[term]["weight"]:
                                entity_candidates[term]["weight"] = float(weight)
                                entity_candidates[term]["topic_id"] = topic_id
                
                # Filter to keep only significant entities
                entities = [e for e in entity_candidates.values() if e["count"] >= 2 and e["weight"] >= 0.01]
            
            # Sort entities by count (descending)
            entities.sort(key=lambda x: x.get("count", 0), reverse=True)
            
            # Return top 20 entities
            return entities[:20]
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {str(e)}")
            return []
    
    def _generate_summary(self, sentence_topics: List[Dict], 
                         lda_model: LdaModel, 
                         num_topics: int) -> str:
        """
        Generate a summary based on key sentences from different topics.
        
        Args:
            sentence_topics: Sentences with topic information
            lda_model: Trained LDA model
            num_topics: Number of topics
            
        Returns:
            Generated summary
        """
        # Get key sentences for summary
        summary_sentences = []
        
        # Get high-probability sentences from different topics
        covered_topics = set()
        
        # First pass: get highest probability sentence from each topic
        for topic_id in range(num_topics):
            topic_sentences = [s for s in sentence_topics if s["dominant_topic"] == topic_id]
            
            if topic_sentences:
                # Sort by probability
                sorted_sentences = sorted(topic_sentences, key=lambda x: x["topic_probability"], reverse=True)
                
                # Get best sentence if probability is significant
                best_sentence = sorted_sentences[0]
                if best_sentence["topic_probability"] > 0.4:
                    summary_sentences.append(best_sentence)
                    covered_topics.add(topic_id)
        
        # Second pass: add important sentences from uncovered topics with lower threshold
        for topic_id in range(num_topics):
            if topic_id in covered_topics:
                continue
                
            topic_sentences = [s for s in sentence_topics if s["dominant_topic"] == topic_id]
            
            if topic_sentences:
                sorted_sentences = sorted(topic_sentences, key=lambda x: x["topic_probability"], reverse=True)
                
                if sorted_sentences and sorted_sentences[0]["topic_probability"] > 0.2:
                    summary_sentences.append(sorted_sentences[0])
        
        # Sort by sentence_id to maintain document flow
        summary_sentences.sort(key=lambda x: x["sentence_id"])
        
        # Create summary with at most 5 sentences
        summary = " ".join([s["sentence"] for s in summary_sentences[:5]])
        
        return summary
    
    def _get_topic_word_matrix(self, lda_model: LdaModel, 
                              dictionary: corpora.Dictionary, 
                              top_n: int = 20) -> List[Dict]:
        """
        Get the topic-word matrix for the Bayesian network.
        
        Args:
            lda_model: Trained LDA model
            dictionary: Word dictionary
            top_n: Number of top words per topic
            
        Returns:
            Topic-word matrix formatted as a list of dicts
        """
        topic_word_matrix = []
        
        for topic_id in range(lda_model.num_topics):
            topic_terms = lda_model.get_topic_terms(topic_id, topn=top_n)
            
            topic_words = {
                "topic_id": topic_id,
                "words": [
                    {"word": dictionary[term_id], "probability": float(prob)}
                    for term_id, prob in topic_terms
                ]
            }
            
            topic_word_matrix.append(topic_words)
        
        return topic_word_matrix
    
    def _calculate_topic_correlations(self, lda_model: LdaModel, corpus: List) -> List[Dict]:
        """
        Calculate correlations between topics for the Bayesian network.
        
        Args:
            lda_model: Trained LDA model
            corpus: Document corpus
            
        Returns:
            Matrix of topic correlations
        """
        # Initialize correlation matrix
        num_topics = lda_model.num_topics
        correlations = np.zeros((num_topics, num_topics))
        
        # Calculate topic distributions for all documents
        all_topics = [lda_model.get_document_topics(doc) for doc in corpus]
        
        # Convert to dense vectors
        topic_vectors = np.zeros((len(all_topics), num_topics))
        
        for i, doc_topics in enumerate(all_topics):
            for topic_id, prob in doc_topics:
                topic_vectors[i, topic_id] = prob
        
        # Calculate correlation matrix (if we have enough documents)
        if len(all_topics) > 2:
            # Calculate correlation matrix, replace NaNs with 0
            raw_correlations = np.corrcoef(topic_vectors.T)
            correlations = np.nan_to_num(raw_correlations)
        
        # Format as list of dicts
        correlation_list = []
        for i in range(num_topics):
            for j in range(i+1, num_topics):  # Only upper triangle
                if abs(correlations[i, j]) > 0.1:  # Only include significant correlations
                    correlation_list.append({
                        "topic_1": i,
                        "topic_2": j,
                        "correlation": float(correlations[i, j])
                    })
        
        # Sort by absolute correlation (descending)
        correlation_list.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return correlation_list 