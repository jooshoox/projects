import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import yaml

# Third-party imports for different file formats
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    from docx import Document
    import chardet
except ImportError as e:
    logging.error(f'Missing required dependency: {e}')
    logging.info('Please install required packages: pip install pdfminer.six python-docx chardet')
    raise

# Set up logging
logger = logging.getLogger(__name__)

class TextExtractor:
    """
    A utility class for extracting text from various file formats.
    Supports PDF, DOC/DOCX, TXT, MD, and RTF files.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the TextExtractor with configuration settings.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.supported_formats = {
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
            '.doc': self._extract_doc,
            '.txt': self._extract_text,
            '.md': self._extract_text,
            '.rtf': self._extract_text
        }

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f'Error loading config: {e}')
            return {}

    def _check_file_size(self, file_path: str) -> bool:
        """
        Check if file size is within configured limits.

        Args:
            file_path (str): Path to the file

        Returns:
            bool: True if file size is acceptable, False otherwise
        """
        max_size = self.config.get('content_analysis', {}).get('thresholds', {}).get('maximum_file_size_mb', 50)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        return file_size_mb <= max_size

    def _extract_pdf(self, file_path: str) -> Optional[str]:
        """
        Extract text from PDF files using pdfminer.six.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            return pdf_extract_text(file_path)
        except Exception as e:
            logger.error(f'Error extracting text from PDF {file_path}: {e}')
            return None

    def _extract_docx(self, file_path: str) -> Optional[str]:
        """
        Extract text from DOCX files using python-docx.

        Args:
            file_path (str): Path to the DOCX file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f'Error extracting text from DOCX {file_path}: {e}')
            return None

    def _extract_doc(self, file_path: str) -> Optional[str]:
        """
        Handle legacy DOC files (placeholder for future implementation).

        Args:
            file_path (str): Path to the DOC file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        logger.warning('Legacy DOC format is not fully supported')
        return None

    def _extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from plain text files with encoding detection.

        Args:
            file_path (str): Path to the text file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            # Detect the file encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Read the file with the detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f'Error extracting text from {file_path}: {e}')
            return None

    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from a file based on its extension.

        Args:
            file_path (str): Path to the file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        file_path = str(Path(file_path))
        extension = Path(file_path).suffix.lower()

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f'File not found: {file_path}')
            return None

        # Check file size
        if not self._check_file_size(file_path):
            logger.warning(f'File too large: {file_path}')
            return None

        # Check if format is supported
        if extension not in self.supported_formats:
            logger.warning(f'Unsupported file format: {extension}')
            return None

        # Extract text using appropriate method
        return self.supported_formats[extension](file_path)

    def get_supported_formats(self) -> list:
        """
        Get list of supported file formats.

        Returns:
            list: List of supported file extensions
        """
        return list(self.supported_formats.keys())
