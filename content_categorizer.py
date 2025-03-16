import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

from text_extractor import TextExtractor
from nlp_analyzer import NLPAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

class ContentCategorizer:
    """
    A class for categorizing files based on their content using NLP analysis.
    Supports parallel processing and caching of results.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the content categorizer with configuration settings.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.text_extractor = TextExtractor(config_path)
        self.nlp_analyzer = NLPAnalyzer(config_path)
        self.processed_files: Set[str] = set()

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f'Error loading config: {e}')
            return {}

    def _create_category_structure(self, base_dir: str) -> None:
        """Create directory structure for categories and tags"""
        try:
            # Create base directories for each tag
            tags = self.config.get('content_analysis', {}).get('tags', {}).keys()
            for tag in tags:
                tag_dir = Path(base_dir) / tag
                tag_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f'Created directory: {tag_dir}')

            # Create unclassified directory
            unclassified_dir = Path(base_dir) / 'Unclassified'
            unclassified_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f'Created directory: {unclassified_dir}')

        except Exception as e:
            logger.error(f'Error creating directory structure: {e}')
            raise

    def _process_file(self, file_path: str, base_dir: str) -> Dict[str, float]:
        """
        Process a single file and return its tag scores.

        Args:
            file_path (str): Path to the file to process
            base_dir (str): Base directory for categorized files

        Returns:
            Dict[str, float]: Dictionary of tag scores
        """
        try:
            # Extract text content
            text = self.text_extractor.extract_text(file_path)
            if text is None:
                logger.warning(f'Could not extract text from {file_path}')
                return {}

            # Detect language (default to English)
            lang = self.config.get('content_analysis', {}).get('languages', {}).get('default', 'en')

            # Analyze content
            scores = self.nlp_analyzer.analyze_text(text, file_path, lang)
            return scores

        except Exception as e:
            logger.error(f'Error processing file {file_path}: {e}')
            return {}

    def _organize_file(self, file_path: str, scores: Dict[str, float], base_dir: str) -> None:
        """
        Organize a file based on its content analysis scores.

        Args:
            file_path (str): Path to the file
            scores (Dict[str, float]): Tag scores for the file
            base_dir (str): Base directory for categorized files
        """
        try:
            file_path = Path(file_path)
            if not scores:
                # Move to unclassified if no matching tags
                dest_dir = Path(base_dir) / 'Unclassified'
                dest_path = dest_dir / file_path.name
                self._safe_file_operation(file_path, dest_path)
                return

            # Handle each matching tag
            create_symlinks = self.config.get('content_analysis', {}).get('processing', {}).get('create_symlinks', True)
            first_tag = True

            for tag, score in scores.items():
                dest_dir = Path(base_dir) / tag
                dest_path = dest_dir / file_path.name

                if first_tag:
                    # Move the file to the first matching category
                    self._safe_file_operation(file_path, dest_path)
                    first_tag = False
                elif create_symlinks:
                    # Create symlinks in other matching categories
                    try:
                        dest_path.symlink_to(file_path)
                        logger.debug(f'Created symlink: {dest_path} -> {file_path}')
                    except Exception as e:
                        logger.warning(f'Could not create symlink {dest_path}: {e}')

        except Exception as e:
            logger.error(f'Error organizing file {file_path}: {e}')

    def _safe_file_operation(self, src: Path, dest: Path) -> None:
        """Safely move or copy a file, handling name conflicts"""
        try:
            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Handle name conflicts
            counter = 1
            final_dest = dest
            while final_dest.exists():
                final_dest = dest.parent / f'{dest.stem}_{counter}{dest.suffix}'
                counter += 1

            # Move or copy based on configuration
            move_files = self.config.get('settings', {}).get('move_files', True)
            if move_files:
                shutil.move(str(src), str(final_dest))
                logger.info(f'Moved: {src} -> {final_dest}')
            else:
                shutil.copy2(str(src), str(final_dest))
                logger.info(f'Copied: {src} -> {final_dest}')

        except Exception as e:
            logger.error(f'Error in file operation {src} -> {dest}: {e}')
            raise

    def categorize_directory(self, directory: str, base_dir: Optional[str] = None) -> None:
        """
        Categorize all supported files in a directory based on their content.

        Args:
            directory (str): Directory containing files to categorize
            base_dir (Optional[str]): Base directory for categorized files
        """
        try:
            directory = Path(directory)
            if not directory.exists():
                raise FileNotFoundError(f'Directory not found: {directory}')

            # Use directory as base_dir if not specified
            base_dir = Path(base_dir or directory)
            self._create_category_structure(base_dir)

            # Get list of files to process
            files = [
                f for f in directory.rglob('*')
                if f.is_file() and f.suffix.lower() in self.text_extractor.get_supported_formats()
            ]

            # Configure parallel processing
            max_workers = self.config.get('content_analysis', {}).get('processing', {}).get('max_workers', 4)
            use_parallel = self.config.get('content_analysis', {}).get('processing', {}).get('parallel_processing', True)

            if use_parallel and len(files) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self._process_file, str(f), str(base_dir)): f
                        for f in files
                    }

                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            scores = future.result()
                            self._organize_file(str(file_path), scores, str(base_dir))
                        except Exception as e:
                            logger.error(f'Error processing {file_path}: {e}')
            else:
                # Process files sequentially
                for file_path in files:
                    try:
                        scores = self._process_file(str(file_path), str(base_dir))
                        self._organize_file(str(file_path), scores, str(base_dir))
                    except Exception as e:
                        logger.error(f'Error processing {file_path}: {e}')

            logger.info(f'Completed content categorization for {len(files)} files')

        except Exception as e:
            logger.error(f'Error during directory categorization: {e}')
            raise

    def get_file_categories(self, file_path: str) -> List[str]:
        """
        Get categories for a single file without moving it.

        Args:
            file_path (str): Path to the file

        Returns:
            List[str]: List of matching categories
        """
        try:
            scores = self._process_file(file_path, '')
            return list(scores.keys())
        except Exception as e:
            logger.error(f'Error getting categories for {file_path}: {e}')
            return []
