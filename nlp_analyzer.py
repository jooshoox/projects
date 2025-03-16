import os
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from collections import Counter
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports for NLP
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError as e:
    logging.error(f'Missing required dependency: {e}')
    logging.info('Please install required packages: pip install nltk')
    raise

# Set up logging
logger = logging.getLogger(__name__)

class NLPAnalyzer:
    """
    A class for analyzing text content using NLP techniques.
    Provides keyword extraction and document classification.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the NLP analyzer with configuration settings.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = {}
        self._initialize_stopwords()

    def _initialize_stopwords(self):
        """Initialize stopwords for all supported languages"""
        supported_langs = self.config.get('content_analysis', {}).get('languages', {}).get('supported', ['en'])
        for lang in supported_langs:
            try:
                self.stopwords[lang] = set(stopwords.words(self._get_nltk_language_name(lang)))
            except Exception as e:
                logger.warning(f'Could not load stopwords for language {lang}: {e}')
                self.stopwords[lang] = set()

    @staticmethod
    def _get_nltk_language_name(lang_code: str) -> str:
        """Convert ISO language code to NLTK language name"""
        language_map = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german'
        }
        return language_map.get(lang_code, 'english')

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f'Error loading config: {e}')
            return {}

    def _get_cache_path(self, file_path: str) -> Path:
        """Generate cache file path for analyzed content"""
        return self.cache_dir / f'{Path(file_path).stem}_analysis.cache'

    def _should_use_cache(self, cache_path: Path, file_path: str) -> bool:
        """
        Determine if cached analysis should be used.
        
        Args:
            cache_path (Path): Path to the cache file
            file_path (str): Path to the original file
            
        Returns:
            bool: True if cache should be used
        """
        if not self.config.get('content_analysis', {}).get('processing', {}).get('enable_caching', True):
            return False

        if not cache_path.exists():
            return False

        cache_duration = self.config.get('content_analysis', {}).get('processing', {}).get('cache_duration_days', 30)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        
        return cache_age.days <= cache_duration

    def analyze_text(self, text: str, file_path: str, lang: str = 'en') -> Dict[str, float]:
        """
        Analyze text content and return tag scores.
        
        Args:
            text (str): Text content to analyze
            file_path (str): Path to the original file (for caching)
            lang (str): Language code
            
        Returns:
            Dict[str, float]: Dictionary of tag scores
        """
        cache_path = self._get_cache_path(file_path)
        
        # Check cache first
        if self._should_use_cache(cache_path, file_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f'Error loading cache for {file_path}: {e}')

        # Perform new analysis
        try:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token.isalnum() and token not in self.stopwords.get(lang, set())]

            # Count token frequencies
            word_freq = Counter(tokens)

            # Calculate scores for each tag
            tag_scores = {}
            tags_config = self.config.get('content_analysis', {}).get('tags', {})
            
            for tag, tag_config in tags_config.items():
                keywords = tag_config.get('keywords', [])
                weight = tag_config.get('weight', 1.0)
                min_confidence = tag_config.get('minimum_confidence', 0.6)
                
                # Calculate tag score based on keyword matches
                score = 0
                for keyword in keywords:
                    keyword = keyword.lower()
                    if keyword in word_freq:
                        score += word_freq[keyword] * weight
                
                # Normalize score
                if score > 0:
                    normalized_score = min(1.0, score / (len(keywords) * weight))
                    if normalized_score >= min_confidence:
                        tag_scores[tag] = normalized_score

            # Cache the results
            if self.config.get('content_analysis', {}).get('processing', {}).get('enable_caching', True):
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(tag_scores, f)
                except Exception as e:
                    logger.warning(f'Error caching analysis for {file_path}: {e}')

            return tag_scores

        except Exception as e:
            logger.error(f'Error analyzing text: {e}')
            return {}

    def get_document_tags(self, text: str, file_path: str, lang: str = 'en') -> List[str]:
        """
        Get list of matching tags for a document.
        
        Args:
            text (str): Text content to analyze
            file_path (str): Path to the original file
            lang (str): Language code
            
        Returns:
            List[str]: List of matching tags
        """
        scores = self.analyze_text(text, file_path, lang)
        return list(scores.keys())

    def get_tag_scores(self, text: str, file_path: str, lang: str = 'en') -> List[Tuple[str, float]]:
        """
        Get sorted list of tag scores.
        
        Args:
            text (str): Text content to analyze
            file_path (str): Path to the original file
            lang (str): Language code
            
        Returns:
            List[Tuple[str, float]]: Sorted list of (tag, score) pairs
        """
        scores = self.analyze_text(text, file_path, lang)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
