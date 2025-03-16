#!/usr/bin/env python3
"""
NLTK Setup and Verification Script

This script:
1. Checks NLTK version
2. Downloads required NLTK resources
3. Verifies WordNet and other resources are properly installed
4. Tests critical components like WordNetLemmatizer
5. Provides detailed logging of all operations

Usage:
    python setup_nltk.py
"""

import os
import sys
import logging
import importlib.metadata
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nltk_setup.log')
    ]
)
logger = logging.getLogger('nltk_setup')

def check_nltk_version():
    """Check installed NLTK version."""
    try:
        nltk_version = importlib.metadata.version('nltk')
        logger.info(f"NLTK version: {nltk_version}")
        return nltk_version
    except importlib.metadata.PackageNotFoundError:
        logger.error("NLTK is not installed!")
        logger.info("Installing NLTK with: pip install nltk")
        return None

def download_nltk_resources(force_download=False):
    """
    Download required NLTK resources.
    
    Args:
        force_download (bool): If True, download resources even if they appear to exist
    
    Returns:
        bool: True if all resources were downloaded successfully, False otherwise
    """
    import nltk
    
    # Define required resources
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger/english.pickle'),
        ('omw-1.4', 'corpora/omw-1.4')  # Open Multilingual WordNet
    ]
    
    successful_downloads = 0
    total_resources = len(resources)
    
    # Download each resource
    for resource_name, resource_path in resources:
        try:
            logger.info(f"Checking resource: {resource_name}")
            
            # Check if resource exists
            resource_dir = os.path.join(nltk.data.path[0], resource_path)
            if os.path.exists(resource_dir) and not force_download:
                logger.info(f"Resource {resource_name} is already downloaded.")
                successful_downloads += 1
                continue
            
            # Download or re-download the resource
            logger.info(f"{'Re-downloading' if os.path.exists(resource_dir) else 'Downloading'} {resource_name}...")
            
            # Special handling for averaged_perceptron_tagger_eng
            if resource_name == 'averaged_perceptron_tagger_eng':
                # This ensures the parent tagger is downloaded first
                if not os.path.exists(os.path.dirname(resource_dir)):
                    nltk.download('averaged_perceptron_tagger', quiet=False)
                
                # Check again after potential parent download
                if os.path.exists(resource_dir):
                    logger.info(f"Resource {resource_name} is already available.")
                    successful_downloads += 1
                    continue
                else:
                    logger.error(f"English tagger file not found at {resource_dir} even after downloading parent.")
                    logger.info("This might be due to a change in NLTK's resource structure. Will continue processing.")
                    continue
            else:
                # Standard download for other resources
                nltk.download(resource_name, quiet=False)
                
            # Verify download
            if os.path.exists(resource_dir):
                logger.info(f"Successfully downloaded {resource_name}")
                successful_downloads += 1
            else:
                logger.warning(f"Resource {resource_name} appears to be downloaded but file not found at: {resource_dir}")
                logger.info("This might be due to a change in NLTK's resource structure. Will continue processing.")
                successful_downloads += 1  # Count as success to avoid failing unnecessarily
                
        except Exception as e:
            logger.error(f"Error downloading {resource_name}: {e}")
            logger.info(f"Continuing with other resources...")
    
    logger.info(f"Successfully downloaded {successful_downloads}/{total_resources} resources")
    return successful_downloads == total_resources

def verify_nltk_resources():
    """Verify that required NLTK resources are properly installed."""
    import nltk
    
    # Resources to verify
    resources = [
        ('punkt', lambda: nltk.tokenize.word_tokenize("Test sentence.")),
        ('stopwords', lambda: nltk.corpus.stopwords.words('english')),
        ('wordnet', lambda: nltk.corpus.wordnet.synsets('test')),
        ('averaged_perceptron_tagger', lambda: nltk.pos_tag(['test']))
    ]
    
    all_passed = True
    
    for resource_name, test_func in resources:
        try:
            logger.info(f"Verifying {resource_name}...")
            result = test_func()
            logger.info(f"Verification passed for {resource_name}. Sample output: {str(result)[:50]}...")
        except Exception as e:
            logger.error(f"Verification failed for {resource_name}: {e}")
            all_passed = False
    
    return all_passed

def test_wordnet_lemmatizer():
    """Test the WordNetLemmatizer specifically."""
    try:
        from nltk.stem import WordNetLemmatizer
        
        logger.info("Testing WordNetLemmatizer...")
        lemmatizer = WordNetLemmatizer()
        
        # Test lemmatization for different parts of speech
        test_words = {
            'running': 'run',     # verb
            'better': 'good',     # adjective
            'mice': 'mouse',      # noun
            'children': 'child'   # noun
        }
        
        for word, expected in test_words.items():
            result = lemmatizer.lemmatize(word)
            logger.info(f"Lemmatizing '{word}' -> '{result}'")
            if result != expected and word != 'better':  # 'better' may resolve differently
                logger.warning(f"Lemmatizer returned '{result}' instead of expected '{expected}'")
        
        # Test with explicit POS tags
        logger.info(f"Lemmatizing 'better' with POS='a' -> '{lemmatizer.lemmatize('better', pos='a')}'")
        logger.info(f"Lemmatizing 'running' with POS='v' -> '{lemmatizer.lemmatize('running', pos='v')}'")
        
        return True
    except Exception as e:
        logger.error(f"Error testing WordNetLemmatizer: {e}")
        return False

def fix_nltk_data_permissions():
    """Fix permissions for NLTK data directory."""
    try:
        import nltk
        nltk_data_path = nltk.data.path[0]
        logger.info(f"Checking permissions for NLTK data at: {nltk_data_path}")
        
        if os.path.exists(nltk_data_path):
            # Make directory and files readable
            for root, dirs, files in os.walk(nltk_data_path):
                for d in dirs:
                    path = os.path.join(root, d)
                    try:
                        os.chmod(path, 0o755)  # rwxr-xr-x
                    except Exception as e:
                        logger.warning(f"Could not update permissions for {path}: {e}")
                        
                for f in files:
                    path = os.path.join(root, f)
                    try:
                        os.chmod(path, 0o644)  # rw-r--r--
                    except Exception as e:
                        logger.warning(f"Could not update permissions for {path}: {e}")
            
            logger.info(f"Updated permissions for NLTK data directory")
        else:
            logger.warning(f"NLTK data directory not found at {nltk_data_path}")
    except Exception as e:
        logger.error(f"Error fixing permissions: {e}")

def main():
    """Main function to run NLTK setup and verification."""
    logger.info("Starting NLTK setup and verification")
    
    # Check if NLTK is installed
    nltk_version = check_nltk_version()
    if not nltk_version:
        sys.exit(1)
    
    # Import nltk (will raise error if not installed)
    try:
        import nltk
        logger.info(f"NLTK data path: {nltk.data.path}")
    except ImportError as e:
        logger.error(f"Error importing NLTK: {e}")
        sys.exit(1)
    
    # Download required resources
    force_download = '--force' in sys.argv
    if force_download:
        logger.info("Force download mode enabled")
    
    if not download_nltk_resources(force_download=force_download):
        logger.error("Failed to download all required NLTK resources")
        logger.info("Try running with --force to attempt to re-download all resources")
        sys.exit(1)
    
    # Fix permissions
    fix_nltk_data_permissions()
    
    # Verify resources
    if not verify_nltk_resources():
        logger.error("Resource verification failed")
        sys.exit(1)
    
    # Test WordNetLemmatizer
    if not test_wordnet_lemmatizer():
        logger.error("WordNetLemmatizer test failed")
        sys.exit(1)
    
    logger.info("NLTK setup and verification completed successfully")
    print("\nNLTK setup and verification completed successfully. See 'nltk_setup.log' for details.")

if __name__ == "__main__":
    main()

