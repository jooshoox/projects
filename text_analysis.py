import os
import re
import json
import logging
import string
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import pickle

# Set up logging
import logging.handlers
import sys

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with INFO level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler with DEBUG level
file_handler = logging.handlers.RotatingFileHandler(
    'text_analysis.log', 
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("Logger initialized with DEBUG level")

# Global flag to track NLTK initialization status
_NLTK_INITIALIZED = False

class TextAnalyzer:
    """
    A lightweight text analysis class that can detect language, extract keywords,
    and categorize text without requiring advanced NLP libraries.
    
    Features:
    - Configuration loading from file or dictionary
    - Simple language detection using character frequency
    - Basic text preprocessing (cleaning, tokenization)
    - Keyword extraction using frequency analysis
    - Category scoring using configurable weights
    - Cache management for analysis results
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize the TextAnalyzer with either a config file path or a config dictionary.
        
        Args:
            config_path: Path to the YAML or JSON configuration file
            config_dict: Dictionary containing configuration values
        """
        self.config = self._load_config(config_path, config_dict)
        self.cache_dir = Path(self.config.get('cache_dir', '.cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize NLTK components
        # Initialize NLTK components
        logger.debug(f"Initializing NLTK components for TextAnalyzer instance with config: {self.config.get('tags', {}).keys()}")
        self.nltk_available = self.initialize_nltk()
        # Language detection data
        self.lang_profiles = self._initialize_language_profiles()
        
        # Load stopwords for supported languages
        self.stopwords = self._load_stopwords()
        
        # Initialize WordNet lemmatizer if available
        self.lemmatizer = None
        if self.nltk_available:
            try:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
                logger.info("WordNet lemmatizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WordNet lemmatizer: {e}")
        
        # Initialize cache expiry
        self.cache_duration = self.config.get('cache_duration_days', 30)
        
        logger.info(f"TextAnalyzer initialized with {len(self.config.get('tags', {}))} tags")
    
    def initialize_nltk(self) -> bool:
        """
        Initialize NLTK components and download required data if not present.
        
        Returns:
            bool: True if NLTK is available and initialized successfully, False otherwise
        """
        global _NLTK_INITIALIZED
        
        # Skip if already initialized
        if _NLTK_INITIALIZED:
            return True
            
        try:
            import nltk
            
            # Define required NLTK resources
            required_resources = [
                ('punkt', 'tokenizers/punkt'),
                ('stopwords', 'corpora/stopwords'),
                ('wordnet', 'corpora/wordnet'),
                ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
            ]
            
            # Check and download missing resources
            for resource_name, resource_path in required_resources:
                try:
                    nltk.data.find(resource_path)
                    logger.debug(f"NLTK resource '{resource_name}' is already available at {resource_path}")
                except LookupError:
                    logger.info(f"Downloading NLTK resource: {resource_name}")
                    try:
                        nltk.download(resource_name, quiet=True)
                        logger.info(f"Successfully downloaded NLTK resource: {resource_name}")
                    except Exception as e:
                        logger.warning(f"Failed to download NLTK resource '{resource_name}': {e}")
                        
            # Verify WordNet is properly loaded
            try:
                from nltk.corpus import wordnet
                synsets = wordnet.synsets('test')
                if not synsets:
                    logger.warning("WordNet appears to be installed but isn't returning results")
                else:
                    logger.info("WordNet is properly initialized")
            except Exception as e:
                logger.warning(f"Error testing WordNet functionality: {e}")
                
            # Verify tokenizer is working
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize("Testing tokenization.")
                if tokens:
                    logger.debug(f"NLTK tokenizer is properly initialized: tokenized 'Testing tokenization.' into {tokens}")
                    logger.info("NLTK tokenizer is properly initialized")
            except Exception as e:
                logger.warning(f"Error testing NLTK tokenizer: {e}")
                
            _NLTK_INITIALIZED = True
            return True
            
        except ImportError:
            logger.warning("NLTK is not available. Basic text processing will be used instead.")
            return False
        except Exception as e:
            logger.warning(f"Error initializing NLTK: {e}")
            return False
    
    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Dict:
        """
        Load configuration from file or dictionary.
        
        Args:
            config_path: Path to the configuration file
            config_dict: Dictionary containing configuration
            
        Returns:
            Dictionary with configuration values
        """
        if config_dict:
            return config_dict
        
        if config_path:
            try:
                # Check file extension to determine format
                if config_path.endswith(('.yaml', '.yml')):
                    try:
                        import yaml
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                            # Extract content_analysis section if present
                            if 'content_analysis' in config:
                                return config['content_analysis']
                            return config
                    except ImportError:
                        logger.warning("YAML module not found. Attempting JSON format.")
                        with open(config_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                else:
                    # Default to JSON
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                
        # Return default configuration
        return {
            'tags': {
                'technical': {
                    'weight': 1.2,
                    'keywords': ['programming', 'software', 'development', 'code', 'technical'],
                    'minimum_confidence': 0.5
                },
                'financial': {
                    'weight': 1.5,
                    'keywords': ['finance', 'budget', 'investment', 'accounting', 'revenue'],
                    'minimum_confidence': 0.5
                }
            },
            'languages': {
                'default': 'en',
                'supported': ['en']
            },
            'thresholds': {
                'minimum_confidence': 0.5,
                'minimum_keyword_frequency': 2
            },
            'cache_duration_days': 30
        }
    
    def _initialize_language_profiles(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize language profiles for detection based on character frequencies.
        
        Returns:
            Dictionary mapping language codes to character frequency profiles
        """
        # Simple language profiles based on character frequency
        profiles = {
            'en': {'e': 0.12, 't': 0.09, 'a': 0.08, 'o': 0.07, 'i': 0.07, 'n': 0.07, 's': 0.06},  # English
            'es': {'e': 0.14, 'a': 0.12, 'o': 0.09, 's': 0.08, 'n': 0.07, 'r': 0.06, 'i': 0.06},  # Spanish
            'fr': {'e': 0.15, 'a': 0.08, 's': 0.08, 'i': 0.07, 'n': 0.07, 't': 0.07, 'r': 0.06},  # French
            'de': {'e': 0.16, 'n': 0.10, 'i': 0.08, 'r': 0.07, 's': 0.07, 't': 0.06, 'a': 0.06}   # German
        }
        
        # Filter to only include supported languages
        supported_langs = self.config.get('languages', {}).get('supported', ['en'])
        return {lang: profiles.get(lang, profiles['en']) for lang in supported_langs}
    
    def _load_stopwords(self) -> Dict[str, Set[str]]:
        """
        Load stopwords for supported languages.
        
        Returns:
            Dictionary mapping language codes to sets of stopwords
        """
        stopwords = {}
        
        # Basic stopwords for English
        en_stopwords = {
            'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 'on', 'with', 
            'as', 'this', 'by', 'be', 'are', 'an', 'not', 'or', 'at', 'from', 'but', 'have',
            'was', 'has', 'had', 'were', 'would', 'could', 'should', 'can', 'may', 'might',
            'will', 'they', 'them', 'their', 'he', 'she', 'his', 'her', 'we', 'you', 'i'
        }
        
        # Add English stopwords by default
        # Add English stopwords by default
        stopwords['en'] = set(en_stopwords)
        
        # Try to load NLTK stopwords if available
        if self.nltk_available:
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'}
                
                for lang_code in self.config.get('languages', {}).get('supported', ['en']):
                    if lang_code in lang_map:
                        try:
                            nltk_lang = lang_map[lang_code]
                            try:
                                # Verify this language's stopwords are available
                                nltk_stopwords.words(nltk_lang)
                            except LookupError:
                                # Try to download this language's stopwords
                                logger.info(f"Downloading stopwords for {nltk_lang}")
                                import nltk
                                nltk.download(f'stopwords', quiet=True)
                                
                            stopwords[lang_code] = set(nltk_stopwords.words(nltk_lang))
                            logger.info(f"Loaded NLTK stopwords for {lang_code}")
                        except Exception as e:
                            logger.warning(f"Failed to load NLTK stopwords for {lang_code}: {e}")
                            # Fall back to English stopwords if we can't load specific language
                            stopwords[lang_code] = stopwords.get('en', set())
            except Exception as e:
                logger.warning(f"Error loading NLTK stopwords: {e}")
                # Use the same English stopwords for all supported languages
                for lang in self.config.get('languages', {}).get('supported', ['en']):
                    stopwords[lang] = set(en_stopwords)
        else:
            logger.warning("NLTK not available. Using basic English stopwords for all languages.")
            # Use the same English stopwords for all supported languages
            for lang in self.config.get('languages', {}).get('supported', ['en']):
                stopwords[lang] = set(en_stopwords)
        return stopwords
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text based on character frequency and additional heuristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr', 'de')
        """
        default_lang = self.config.get('languages', {}).get('default', 'en')
        
        if not text or len(text) < 50:
            return default_lang
            
        # Create frequency profile of the text
        text = text.lower()
        char_count = Counter(c for c in text if c in string.ascii_lowercase)
        total_chars = sum(char_count.values())
        
        if total_chars == 0:
            return default_lang
            
        text_profile = {char: count/total_chars for char, count in char_count.items()}
        
        # Language-specific word patterns for additional verification
        language_patterns = {
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'that', 'for'],
            'es': ['el', 'la', 'los', 'las', 'y', 'en', 'que', 'es', 'por'],
            'fr': ['le', 'la', 'les', 'et', 'en', 'que', 'est', 'dans', 'pour'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'von', 'mit', 'zu', 'für']
        }
        
        # Count occurrences of language-specific patterns
        pattern_scores = {lang: 0 for lang in language_patterns}
        words = re.findall(r'\b\w+\b', text)
        
        for lang, patterns in language_patterns.items():
            if lang in self.lang_profiles:  # Only check supported languages
                lang_pattern_count = sum(words.count(pattern) for pattern in patterns)
                if words:
                    pattern_scores[lang] = lang_pattern_count / len(words)
        
        # Compare frequency profiles to find the best match
        profile_scores = {}
        for lang, profile in self.lang_profiles.items():
            score = 0
            matched_chars = 0
            for char, freq in profile.items():
                if char in text_profile:
                    # Calculate how close the frequencies are
                    score += 1 - abs(text_profile[char] - freq) / max(text_profile[char], freq)
                    matched_chars += 1
            
            # Weight score by how many characters actually matched
            char_match_ratio = matched_chars / len(profile)
            profile_scores[lang] = (score / len(profile)) * char_match_ratio
        
        # Combine frequency profile scores with pattern scores for languages that have both
        combined_scores = {}
        for lang in self.lang_profiles:
            if lang in pattern_scores:
                # Weight profile scores more heavily (70%) than pattern scores (30%)
                combined_scores[lang] = (profile_scores[lang] * 0.7) + (pattern_scores.get(lang, 0) * 0.3)
            else:
                combined_scores[lang] = profile_scores[lang]
        
        # Find the language with the best combined score
        best_score = 0
        best_lang = default_lang
        for lang, score in combined_scores.items():
            if score > best_score:
                best_score = score
                best_lang = lang
        
        # If the best score is below a threshold, default to configured language
        if best_score < 0.4:  # Minimum confidence for language detection
            logger.debug(f"Language detection score {best_score} below threshold, defaulting to {default_lang}")
            return default_lang
            
        logger.debug(f"Detected language {best_lang} with score {best_score}")
        return best_lang
    
    def preprocess_text(self, text: str, language: str = 'en') -> List[str]:
        """
        Clean and tokenize text for analysis.
        
        Args:
            text: Input text to process
            language: Language code
            
        Returns:
            List of cleaned tokens
        """
        # Validate input text
        if not text:
            logger.warning("Empty text provided for preprocessing")
            return []
            
        if not isinstance(text, str):
            logger.error(f"Invalid input type for preprocessing: {type(text)}")
            return []
            
        logger.debug(f"Starting text preprocessing for language: {language}")
        logger.debug(f"Input text length: {len(text)} characters")
        
        # Convert to lowercase
        text = text.lower()
        logger.debug("Text converted to lowercase")
        
        # Remove URLs, email addresses, and special characters
        original_length = len(text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        logger.debug(f"Removed URLs, emails, and special characters. Length change: {original_length} -> {len(text)}")
        
        # Check if text is still valid after cleaning
        if not text.strip():
            logger.warning("Text is empty after cleaning")
            return []
        
        # Tokenize by whitespace
        tokens = text.split()
        logger.debug(f"Tokenized text into {len(tokens)} tokens")
        
        # Remove stopwords
        stopwords_set = self.stopwords.get(language, self.stopwords.get('en', set()))
        original_token_count = len(tokens)
        tokens = [token for token in tokens if token not in stopwords_set]
        logger.debug(f"Removed stopwords. Token count: {original_token_count} -> {len(tokens)}")
        
        # Remove very short tokens
        original_token_count = len(tokens)
        tokens = [token for token in tokens if len(token) > 2]
        logger.debug(f"Removed short tokens. Token count: {original_token_count} -> {len(tokens)}")
        
        # Check if we have any tokens left
        if not tokens:
            logger.warning("No valid tokens remain after preprocessing")
            return []
        
        # Try to use NLTK's WordNet lemmatizer if available
        if self.lemmatizer and language == 'en':
            try:
                # Try to use NLTK's part-of-speech tagger to improve lemmatization
                try:
                    from nltk import pos_tag
                    from nltk.tokenize import word_tokenize
                    
                    logger.debug("Attempting POS-aware lemmatization")
                    # Get POS tags
                    tagged_tokens = pos_tag(tokens)
                    
                    # Convert Penn Treebank tags to WordNet POS tags
                    lemmatized = []
                    for word, tag in tagged_tokens:
                        wn_tag = 'n'  # Default to noun
                        if tag.startswith('J'):
                            wn_tag = 'a'  # Adjective
                        elif tag.startswith('V'):
                            wn_tag = 'v'  # Verb
                        elif tag.startswith('R'):
                            wn_tag = 'r'  # Adverb
                            
                        lemmatized.append(self.lemmatizer.lemmatize(word, pos=wn_tag))
                    tokens = lemmatized
                    logger.debug("Used POS-aware lemmatization successfully")
                    
                except Exception as e:
                    # Fall back to basic lemmatization without POS
                    logger.debug(f"POS tagging failed, using basic lemmatization: {e}")
                    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                    logger.debug("Used basic lemmatization without POS tagging")
                    
            except Exception as e:
                logger.warning(f"Error during lemmatization: {e}")
                logger.debug("Continuing with unlemmatized tokens")
                # Just continue with the existing tokens - no stems defined here
        
        logger.debug(f"Preprocessing complete. Final token count: {len(tokens)}")
        logger.debug(f"Preprocessing complete. Final token count: {len(tokens)}")
        return tokens
    def _get_cache_path(self, file_path: str) -> Path:
        """
        Generate the path for a cache file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Path object for the cache file
        """
        # Create a hash of the path to avoid filename issues
        file_hash = str(hash(file_path))
        cache_file = self.cache_dir / f"analysis_{file_hash}.cache"
        logger.debug(f"Cache path for {file_path} calculated as {cache_file}")
        return cache_file
    
    def _should_use_cache(self, cache_path: Path, file_path: str) -> bool:
        """
        Determine if cached analysis should be used.
        
        Args:
            cache_path: Path to cache file
            file_path: Original file path
            
        Returns:
            True if cache should be used, False otherwise
        """
        if not self.config.get('enable_caching', True):
            return False
            
        if not cache_path.exists():
            return False
            
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=self.cache_duration):
            return False
            
        # Check if original file has been modified
        try:
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            cache_modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_modified_time > cache_modified_time:
                return False
        except (FileNotFoundError, PermissionError):
            return False
            
        return True
    
    def extract_keywords(self, tokens: List[str], max_keywords: int = 20) -> List[Tuple[str, int]]:
        """
        Extract important keywords from tokenized text, including n-grams.
        
        Args:
            tokens: List of preprocessed tokens
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of (keyword, frequency) tuples
        """
        # Count single token frequencies
        token_freq = Counter(tokens)
        
        # Generate and count bi-grams (pairs of adjacent words)
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(tokens[i] + " " + tokens[i+1])
        bigram_freq = Counter(bigrams)
        
        # Generate and count tri-grams (triplets of adjacent words)
        trigrams = []
        for i in range(len(tokens) - 2):
            trigrams.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2])
        trigram_freq = Counter(trigrams)
        
        # Combine frequencies with appropriate weighting
        min_freq = self.config.get('thresholds', {}).get('minimum_keyword_frequency', 2)
        
        # Single words: standard weight
        keywords = [(word, freq) for word, freq in token_freq.items() if freq >= min_freq]
        
        # Bi-grams: higher weight for multi-word matches
        bigram_min = max(2, min_freq - 1)  # Slightly lower threshold for n-grams
        for bigram, freq in bigram_freq.items():
            if freq >= bigram_min:
                keywords.append((bigram, freq * 1.5))  # Weight bi-grams 1.5x
        
        # Tri-grams: highest weight
        trigram_min = max(2, min_freq - 2)  # Even lower threshold for tri-grams
        for trigram, freq in trigram_freq.items():
            if freq >= trigram_min:
                keywords.append((trigram, freq * 2.0))  # Weight tri-grams 2x
        
        # Apply term frequency-inverse document frequency style weighting
        # Words that appear in many contexts get lower weights
        word_uniqueness = {}
        for word, freq in token_freq.items():
            # Calculate uniqueness score based on how concentrated the word usage is
            positions = [i for i, t in enumerate(tokens) if t == word]
            if len(positions) <= 1:
                uniqueness = 1.0
            else:
                avg_gap = sum(positions[i+1] - positions[i] for i in range(len(positions)-1)) / (len(positions)-1)
                uniqueness = min(1.0, avg_gap / 20)  # Normalize, higher gaps mean more uniqueness
            word_uniqueness[word] = 0.5 + (0.5 * uniqueness)  # Scale between 0.5 and 1.0
        
        # Apply uniqueness weighting to single-word keywords
        for i, (word, freq) in enumerate(keywords):
            if " " not in word and word in word_uniqueness:
                keywords[i] = (word, freq * word_uniqueness[word])
        
        # Sort by weighted frequency (descending)
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords[:max_keywords]
    def score_categories(self, keywords: List[Tuple[str, int]], 
                         language: str = 'en') -> Dict[str, float]:
        """
        Score document categories based on keyword matches with improved matching algorithms.
        
        Args:
            keywords: List of (keyword, frequency) tuples
            language: Language code
            
        Returns:
            Dictionary of category names to confidence scores
        """
        logger.debug(f"Scoring categories for {len(keywords)} keywords in language '{language}'")
        scores = {}
        keyword_dict = dict(keywords)
        
        # Get the maximum frequency for normalization
        max_freq = max(keyword_dict.values()) if keyword_dict else 1
        total_freq = sum(keyword_dict.values())
        
        # Calculate keyword density (importance relative to document)
        keyword_importance = {}
        for kw, freq in keyword_dict.items():
            # Calculate importance based on frequency and term length
            # Longer terms get a boost as they are typically more specific
            length_factor = 1.0 + (min(len(kw.split()), 3) - 1) * 0.2
            importance = (freq / total_freq) * length_factor
            keyword_importance[kw] = importance
        
        # Generate variations of multi-word keywords for better matching
        compound_variations = {}
        for kw, freq in keyword_dict.items():
            if " " in kw:
                # Add the original multi-word term with full weight
                compound_variations[kw] = freq
                
                # Add variations with word order permutations (for 2-3 word terms)
                parts = kw.split()
                if len(parts) == 2:
                    variation = f"{parts[1]} {parts[0]}"
                    compound_variations[variation] = freq * 0.8  # 80% weight for reordered terms
                
                # Add individual parts with reduced weight
                for part in parts:
                    if len(part) > 3:  # Only consider meaningful parts
                        compound_variations[part] = compound_variations.get(part, 0) + (freq * 0.5)
        
        # Merge variations back into the main dictionary (if they don't already exist)
        for kw, freq in compound_variations.items():
            if kw not in keyword_dict:
                keyword_dict[kw] = freq
        
        # Calculate scores for each tag
        tags_config = self.config.get('tags', {})
        
        for tag, tag_config in tags_config.items():
            tag_keywords = tag_config.get('keywords', [])
            weight = tag_config.get('weight', 1.0)
            min_confidence = tag_config.get('minimum_confidence', 0.6)
            position_boost = tag_config.get('position_boost', 1.2)  # Boost for keywords at the beginning
            
            # Skip empty keyword lists
            if not tag_keywords:
                continue
                
            # Calculate raw score
            tag_score = 0
            matches = 0
            matched_keywords = set()
            keyword_weights = []  # Store weights for normalization
            
            # First pass: Check for exact matches of individual keywords
            for keyword in tag_keywords:
                keyword = keyword.lower()
                keyword_weights.append(weight)  # Track keyword weights for normalization
                
                # Check if this is a multi-word keyword
                if " " in keyword:
                    keyword_parts = keyword.split()
                    
                    # Full phrase match (highest value)
                    if keyword in keyword_dict:
                        # Boost exact matches with higher weight (2.0x instead of 1.5x)
                        importance_factor = keyword_importance.get(keyword, 1.0) * 1.2
                        score_value = (keyword_dict[keyword] / max_freq) * weight * 2.0 * importance_factor
                        tag_score += score_value
                        matches += 1.2  # Increased match value for exact phrase matches
                        matched_keywords.add(keyword)
                        
                    # Subphrase match (e.g. "data science" in "applied data science")
                    elif any(keyword in kw for kw in keyword_dict if " " in kw):
                        containing_phrases = [kw for kw in keyword_dict if " " in kw and keyword in kw]
                        best_score = 0
                        for phrase in containing_phrases:
                            phrase_score = (keyword_dict[phrase] / max_freq) * weight * 1.7
                            best_score = max(best_score, phrase_score)
                        
                        if best_score > 0:
                            tag_score += best_score
                            matches += 1.0
                            matched_keywords.add(keyword)
                            
                    # Partial phrase match (check if all parts exist)
                    elif all(part in keyword_dict for part in keyword_parts):
                        part_scores = [(keyword_dict[part] / max_freq) * keyword_importance.get(part, 1.0) 
                                      for part in keyword_parts if part in keyword_dict]
                        avg_score = sum(part_scores) / len(part_scores) if part_scores else 0
                        tag_score += avg_score * weight * 1.2  # Increased from 1.0 to 1.2
                        matches += 0.9  # Increased from 0.8 to 0.9
                        matched_keywords.update(keyword_parts)
                        
                    # Some parts match (more meaningful if longer parts match)
                    else:
                        matching_parts = [part for part in keyword_parts if part in keyword_dict]
                        if matching_parts:
                            # Weight by length of matching parts
                            length_weights = [0.8 + min(len(part), 10) * 0.03 for part in matching_parts]
                            part_scores = [(keyword_dict[part] / max_freq) * length_weights[i] * keyword_importance.get(part, 1.0)
                                          for i, part in enumerate(matching_parts)]
                            avg_score = sum(part_scores) / len(part_scores)
                            match_ratio = len(matching_parts) / len(keyword_parts)
                            
                            # Adjust weight based on which parts match (nouns and verbs matter more)
                            meaningful_boost = 1.0
                            if any(len(part) > 5 for part in matching_parts):  # Longer words likely more meaningful
                                meaningful_boost = 1.2
                                
                            tag_score += avg_score * weight * 0.8 * match_ratio * meaningful_boost
                            matches += 0.6 * match_ratio  # Increased from 0.5 to 0.6
                            matched_keywords.update(matching_parts)
                else:
                    # Single word exact match
                    if keyword in keyword_dict:
                        importance_factor = keyword_importance.get(keyword, 1.0) * 1.5  # Higher importance for exact matches
                        score_value = (keyword_dict[keyword] / max_freq) * weight * importance_factor
                        
                        # Apply position boost if this is a high-frequency keyword
                        if keyword_dict[keyword] > max_freq * 0.6:  # Reduced threshold from 0.7 to 0.6
                            score_value *= position_boost
                            
                        tag_score += score_value
                        matches += 1
                        matched_keywords.add(keyword)
                        
            # Calculate final score and normalize
            if len(tag_keywords) > 0 and matches > 0:
                # Calculate average weight for normalization
                avg_weight = sum(keyword_weights) / len(keyword_weights)
                
                # Normalize score based on number of keywords and their weights
                max_possible_score = len(tag_keywords) * avg_weight * 2.0  # 2.0 is the maximum weight multiplier
                
                # Apply normalization
                normalized_score = min(1.0, tag_score / max_possible_score)
                
                # Apply importance based on how many keywords matched
                match_ratio = matches / len(tag_keywords)
                match_importance = 0.5 + (0.5 * match_ratio)  # Scale between 0.5 and 1.0
                
                # Weighted score combining normalized score and match coverage
                final_score = normalized_score * match_importance
                
                # Apply tag-specific threshold
                if final_score >= min_confidence:
                    scores[tag] = final_score
                    
                    # Log score components for debugging
                    logger.debug(f"Category '{tag}' score: {final_score:.2f}")
                    logger.debug(f"  - Raw score: {tag_score:.2f}")
                    logger.debug(f"  - Normalized score: {normalized_score:.2f}")
                    logger.debug(f"  - Match ratio: {match_ratio:.2f}")
                    logger.debug(f"  - Matched keywords: {', '.join(matched_keywords)}")
                    logger.debug(f"  - Tag threshold: {min_confidence}")
                else:
                    logger.debug(f"Category '{tag}' score {final_score:.2f} below threshold {min_confidence}")
        
        # Apply global threshold after all categories are scored
        global_threshold = self.config.get('thresholds', {}).get('minimum_confidence', 0.6)
        scores = {tag: score for tag, score in scores.items() if score >= global_threshold}
        
        # If we have scores, ensure they sum to 1.0 for relative comparison
        if scores:
            # Log final category scores
            logger.info(f"Final category scores: {', '.join([f'{t}:{s:.2f}' for t, s in scores.items()])}")
        else:
            logger.info("No categories met the confidence threshold")
            
        return scores

    def cache_result(self, file_path: str, analysis_result: Dict[str, Any]) -> bool:
        """
        Cache analysis results for a file.
        
        Args:
            file_path: Path to the original file
            analysis_result: Analysis results to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self._get_cache_path(file_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis_result, f)
            logger.debug(f"Cached analysis results for {file_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache analysis results for {file_path}: {e}")
            return False
    
    def load_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load cached analysis results for a file.
        
        Args:
            file_path: Path to the original file
            
        Returns:
            Cached analysis results or None if cache doesn't exist or is invalid
        """
        cache_path = self._get_cache_path(file_path)
        
        if not self._should_use_cache(cache_path, file_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_path}: {e}")
            return None
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect the encoding of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding or 'utf-8' if detection fails
        """
        try:
            import chardet
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                sample = f.read(min(1024 * 1024, os.path.getsize(file_path)))  # Read up to 1MB
            result = chardet.detect(sample)
            encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
            return encoding
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path}: {e}")
            return 'utf-8'  # Default to UTF-8
    
    def _extract_text_from_file(self, file_path: str) -> Optional[str]:
        """
        Extract text from various file types.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text or None if extraction fails
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Plain text files
        if file_ext in ['.txt', '.md', '.csv', '.log', '.json', '.xml', '.html', '.htm', '.css', '.js', '.py', '.java', '.c', '.cpp', '.h', '.sh', '.bat']:
            try:
                encoding = self._detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return None
        
        # PDF files
        elif file_ext == '.pdf':
            try:
                # Try to use pdfminer.six if available
                from pdfminer.high_level import extract_text
                return extract_text(file_path)
            except ImportError:
                logger.warning("pdfminer.six not available. Trying alternative PDF extraction method.")
                try:
                    # Try PyPDF2 as a fallback
                    import PyPDF2
                    text = []
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            text.append(page.extract_text())
                    return "\n".join(text)
                except ImportError:
                    logger.error("No PDF extraction library available. Install pdfminer.six or PyPDF2.")
                    return None
                except Exception as e:
                    logger.error(f"Error extracting text from PDF {file_path}: {e}")
                    return None
        
        # Microsoft Word documents
        elif file_ext in ['.docx', '.doc']:
            try:
                # Try python-docx for DOCX files
                if file_ext == '.docx':
                    from docx import Document
                    doc = Document(file_path)
                    return "\n".join(paragraph.text for paragraph in doc.paragraphs)
                # For DOC files, we can use other libraries or convert them first
                else:
                    # Try textract as a general-purpose extraction tool
                    try:
                        import textract
                        return textract.process(file_path).decode('utf-8')
                    except ImportError:
                        logger.warning("textract not available for DOC extraction.")
                        return None
            except ImportError:
                logger.error("Required library for Word document extraction not available.")
                return None
            except Exception as e:
                logger.error(f"Error extracting text from Word document {file_path}: {e}")
                return None
        
        # RTF files
        elif file_ext == '.rtf':
            try:
                # Try striprtf
                from striprtf.striprtf import rtf_to_text
                encoding = self._detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    rtf_text = f.read()
                return rtf_to_text(rtf_text)
            except ImportError:
                logger.warning("striprtf not available for RTF extraction.")
                try:
                    # Fallback to textract
                    import textract
                    return textract.process(file_path).decode('utf-8')
                except ImportError:
                    logger.error("No RTF extraction library available.")
                    return None
            except Exception as e:
                logger.error(f"Error extracting text from RTF {file_path}: {e}")
                return None
        
        # Unsupported file type
        else:
            logger.warning(f"Unsupported file type for text extraction: {file_ext}")
            return None
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Perform complete analysis of a file, extracting and analyzing text content
        from various file formats including PDF, DOCX, and plain text.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with analysis results including:
            - extracted text
            - file metadata
            - language detection
            - keywords
            - category scores
            - analysis timestamp
        """
        # Validate input
        if not file_path:
            logger.error("Empty file path provided for analysis")
            return {
                'error': "Empty file path provided",
                'analyzed_at': datetime.now().isoformat(),
                'status': 'error'
            }
            
        logger.debug(f"===== STARTING ANALYSIS OF FILE: {file_path} =====")
        logger.info(f"Starting analysis of file: {file_path}")
            
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {
                'error': f"File not found: {file_path}",
                'file_path': file_path,
                'analyzed_at': datetime.now().isoformat(),
                'status': 'error'
            }
        # Check file size limit
        max_size_mb = self.config.get('thresholds', {}).get('maximum_file_size_mb', 50)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return {
                'error': f"File exceeds maximum size limit of {max_size_mb}MB",
                'file_path': file_path,
                'file_size_mb': file_size_mb,
                'analyzed_at': datetime.now().isoformat(),
                'status': 'error'
            }
        
        # Check cache first
        cached_result = self.load_cache(file_path)
        if cached_result:
            logger.info(f"Using cached analysis for {file_path}")
            cached_result['status'] = 'cached'
            return cached_result
        
        # Get file metadata
        file_info = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_extension': os.path.splitext(file_path)[1].lower(),
            'file_size_bytes': os.path.getsize(file_path),
            'file_size_mb': file_size_mb,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'analyzed_at': datetime.now().isoformat(),
            'status': 'analyzed'
        }
        
        try:
            # Extract text content from file
            logger.debug(f"Extracting text content from {file_path} (extension: {file_info['file_extension']})")
            content = self._extract_text_from_file(file_path)
            
            if content is None:
                logger.error(f"Failed to extract any text from {file_path}")
                return {
                    **file_info,
                    'error': "Could not extract text from file (unsupported format or extraction error)",
                    'status': 'error'
                }
                
            if not content.strip():
                logger.warning(f"Extracted text from {file_path} is empty")
                return {
                    **file_info,
                    'error': "Extracted text is empty - file appears to contain no text content",
                    'status': 'error'
                }
                
            logger.info(f"Successfully extracted {len(content)} characters from {file_path}")
            logger.debug(f"Content sample: {content[:100]}...")
            
            # Store content sample (first 1000 chars)
            content_sample = content[:1000] + ('...' if len(content) > 1000 else '')
            
            # Detect language
            language = self.detect_language(content)
            
            # Preprocess text
            logger.info(f"Preprocessing text content (language: {language})")
            logger.debug(f"Detected language: {language} for file {file_path}")
            tokens = self.preprocess_text(content, language)
            
            # Validate tokens after preprocessing
            if not tokens:
                logger.error(f"No valid tokens remain after preprocessing text from {file_path}")
                return {
                    **file_info,
                    'error': "Text preprocessing yielded no valid tokens - content may be invalid or in an unsupported language",
                    'content_sample': content_sample,
                    'language': language,
                    'status': 'error'
                }
                
            logger.info(f"Text preprocessing complete: {len(tokens)} tokens extracted")
            logger.debug(f"Sample tokens: {tokens[:20] if len(tokens) > 20 else tokens}")
            
            # Extract keywords
            logger.info(f"Extracting keywords from processed tokens")
            keywords = self.extract_keywords(tokens)
            logger.info(f"Extracted {len(keywords)} keywords")
            logger.debug(f"Top keywords: {keywords[:10] if len(keywords) > 10 else keywords}")
            
            # Score categories
            logger.debug(f"Scoring categories based on extracted keywords")
            category_scores = self.score_categories(keywords, language)
            logger.debug(f"Category scores: {category_scores}")
            
            # Calculate content statistics
            word_count = len(content.split())
            sentence_count = len(re.split(r'[.!?]+', content))
            
            # Prepare result
            result = {
                **file_info,
                'language': language,
                'keywords': keywords,
                'categories': category_scores,
                'content_sample': content_sample,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'token_count': len(tokens)
            }
            
            # Cache result
            # Cache result
            logger.debug(f"Caching analysis result for {file_path}")
            self.cache_result(file_path, result)
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing file {file_path}: {error_msg}")
            logger.debug(f"Exception details for {file_path}: {type(e).__name__} - {str(e)}", exc_info=True)
            return {
                **file_info,
                'error': error_msg,
                'status': 'error'
            }
    
    def get_categories(self, analysis_result: Dict[str, Any], threshold: Optional[float] = None) -> List[str]:
        """
        Get categories that meet or exceed the confidence threshold.
        
        Args:
            analysis_result: Analysis result from analyze_file
            threshold: Minimum confidence threshold (overrides config)
            
        Returns:
            List of category names that meet the threshold
        """
        if 'categories' not in analysis_result:
            return []
            
        categories = analysis_result['categories']
        
        # Use provided threshold or fall back to config
        if threshold is None:
            threshold = self.config.get('thresholds', {}).get('minimum_confidence', 0.6)
            
        # Filter categories by threshold
        return [category for category, score in categories.items() if score >= threshold]
    
    def format_results(self, analysis_result: Dict[str, Any], include_keywords: bool = True) -> str:
        """
        Format analysis results as a human-readable string.
        
        Args:
            analysis_result: Analysis result from analyze_file
            include_keywords: Whether to include keywords in the output
            
        Returns:
            Formatted string with analysis results
        """
        if 'error' in analysis_result:
            return f"Error analyzing {analysis_result.get('file_path', 'file')}: {analysis_result['error']}"
            
        output = []
        output.append(f"Analysis of: {analysis_result.get('file_path', 'text')}")
        output.append(f"Language detected: {analysis_result.get('language', 'unknown')}")
        output.append(f"Analyzed at: {analysis_result.get('analyzed_at', datetime.now().isoformat())}")
        
        categories = analysis_result.get('categories', {})
        if categories:
            output.append("\nCategories:")
            for category, score in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                output.append(f"  - {category}: {score:.2f}")
        else:
            output.append("\nNo matching categories found.")
            
        if include_keywords and 'keywords' in analysis_result:
            keywords = analysis_result['keywords']
            if keywords:
                output.append("\nTop keywords:")
                for keyword, freq in keywords:
                    output.append(f"  - {keyword}: {freq}")
                    
        return "\n".join(output)
    
    def batch_analyze_files(self, file_paths: List[str], parallel: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple files in batch, optionally using parallel processing.
        
        Args:
            file_paths: List of file paths to analyze
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary mapping file paths to their analysis results
        """
        results = {}
        
        if parallel and len(file_paths) > 1:
            try:
                from concurrent.futures import ThreadPoolExecutor
                max_workers = min(os.cpu_count() or 4, 8)
                logger.debug(f"Starting parallel analysis with {max_workers} workers for {len(file_paths)} files")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {executor.submit(self.analyze_file, file_path): file_path for file_path in file_paths}
                    for future in future_to_file:
                        file_path = future_to_file[future]
                        try:
                            results[file_path] = future.result()
                        except Exception as e:
                            logger.error(f"Error in parallel analysis of {file_path}: {e}")
                            results[file_path] = {'error': str(e), 'file_path': file_path}
            except ImportError:
                logger.warning("ThreadPoolExecutor not available. Falling back to sequential processing.")
                for file_path in file_paths:
                    results[file_path] = self.analyze_file(file_path)
        else:
            for file_path in file_paths:
                results[file_path] = self.analyze_file(file_path)
                
        return results
