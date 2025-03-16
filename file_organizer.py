#!/usr/bin/env python3
import os
import shutil
import argparse
import logging
import yaml
import datetime
import time
import math
from pathlib import Path

# Import content analysis modules
try:
    from text_analysis import TextAnalyzer
    CONTENT_ANALYSIS_AVAILABLE = True
except ImportError:
    CONTENT_ANALYSIS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('file_organizer')

# Check if required packages are installed for content analysis
def check_content_analysis_dependencies():
    """
    Check if required packages for content analysis are installed
    
    Returns:
        bool: True if all required packages are installed, False otherwise
    """
    if not CONTENT_ANALYSIS_AVAILABLE:
        logger.warning("TextAnalyzer module not found. Content analysis features will be unavailable.")
        logger.info("To enable content analysis, install required packages: pip install -r requirements.txt")
        return False
    
    # Check additional dependencies that might be needed
    try:
        import yaml
        return True
    except ImportError:
        logger.warning("PyYAML not found. Required for content analysis configuration.")
        logger.info("Please install with: pip install pyyaml")
        return False

# Dictionary mapping file extensions to folder names
EXTENSION_MAP = {
    # Documents
    '.pdf': 'Documents/PDFs',
    '.doc': 'Documents/Word',
    '.docx': 'Documents/Word',
    '.txt': 'Documents/Text',
    '.rtf': 'Documents/Text',
    '.odt': 'Documents/Text',
    
    # Images
    '.jpg': 'Images/JPEG',
    '.jpeg': 'Images/JPEG',
    '.png': 'Images/PNG',
    '.gif': 'Images/GIF',
    '.bmp': 'Images/BMP',
    '.svg': 'Images/SVG',
    
    # Audio
    '.mp3': 'Audio/MP3',
    '.wav': 'Audio/WAV',
    '.flac': 'Audio/FLAC',
    '.aac': 'Audio/AAC',
    
    # Video
    '.mp4': 'Video/MP4',
    '.avi': 'Video/AVI',
    '.mkv': 'Video/MKV',
    '.mov': 'Video/MOV',
    
    # Archives
    '.zip': 'Archives/ZIP',
    '.rar': 'Archives/RAR',
    '.tar': 'Archives/TAR',
    '.gz': 'Archives/GZIP',
    
    # Executables
    '.exe': 'Executables/Windows',
    '.msi': 'Executables/Windows',
    '.app': 'Executables/MacOS',
    '.sh': 'Executables/Scripts',
    '.bat': 'Executables/Scripts',
    
    # Code
    '.py': 'Code/Python',
    '.js': 'Code/JavaScript',
    '.html': 'Code/HTML',
    '.css': 'Code/CSS',
    '.java': 'Code/Java',
    '.c': 'Code/C',
    '.cpp': 'Code/C++',
}

def human_readable_size(size_bytes):
    """
    Convert size in bytes to human-readable format
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Human-readable size (e.g., "1.23 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def load_config(config_file="config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary with extension map and settings
    """
    default_config = {
        "categories": {
            "documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"],
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
            "audio": [".mp3", ".wav", ".flac", ".m4a", ".ogg"],
            "video": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
            "archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
            "code": [".py", ".js", ".java", ".cpp", ".h", ".css", ".html"]
        },
        "settings": {
            "create_missing_dirs": True,
            "dry_run": False,
            "minimum_file_size": 0,
            "ignore_older_than": 0,
            "move_files": True
        }
    }
    
    try:
        if os.path.exists(config_file):
            logger.info(f"Loading configuration from {config_file}")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded configuration: {config}")
                return config
        else:
            logger.warning(f"Configuration file {config_file} not found, using default configuration")
            return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
        return default_config

def build_extension_map(config):
    """
    Build extension map from configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Extension map dictionary
    """
    extension_map = {}
    
    try:
        # Use the config file categories if available
        if "categories" in config:
            for category, extensions in config["categories"].items():
                for ext in extensions:
                    # Ensure extension starts with a dot
                    if not ext.startswith('.'):
                        ext = f".{ext}"
                    extension_map[ext.lower()] = f"{category.capitalize()}"
        else:
            # Fall back to the default extension map
            extension_map = EXTENSION_MAP
            
        logger.debug(f"Built extension map: {extension_map}")
        return extension_map
    except Exception as e:
        logger.error(f"Error building extension map: {e}")
        # Fall back to the default extension map
        return EXTENSION_MAP

def create_directories(base_dir, extension_map=None):
    """
    Create the necessary directories based on the extension map
    
    Args:
        base_dir (str): Base directory where folders will be created
        extension_map (dict, optional): Map of extensions to directories
    """
    try:
        # Use provided extension map or default to EXTENSION_MAP
        map_to_use = extension_map or EXTENSION_MAP
        
        # Create a set of unique directory paths from the extension map
        unique_dirs = set(map_to_use.values())
        
        for dir_path in unique_dirs:
            full_path = os.path.join(base_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
        
        # Create an "Other" folder for unknown extensions
        other_folder = os.path.join(base_dir, "Other")
        os.makedirs(other_folder, exist_ok=True)
        
        logger.info(f"Created all necessary directories in {base_dir}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def get_unique_filename(target_path):
    """
    Generate a unique filename if a file with the same name already exists
    
    Args:
        target_path (str): Target file path
        
    Returns:
        str: Unique file path
    """
    if not os.path.exists(target_path):
        return target_path
    
    directory, filename = os.path.split(target_path)
    name, extension = os.path.splitext(filename)
    
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        if not os.path.exists(new_path):
            return new_path
        
        counter += 1

def organize_files(directory, config=None, args=None):
    """
    Organize files in the specified directory based on their extensions
    or content analysis.
    
    Args:
        directory (str): Directory to organize
        config (dict, optional): Configuration dictionary
        args (argparse.Namespace, optional): Command line arguments
        
    Returns:
        bool: True if organization was successful, False otherwise
    """
    try:
        # Use default empty config if none provided
        if config is None:
            config = {}
            
        # Extract settings from config with defaults
        settings = config.get("settings", {})
        dry_run = args.dry_run if args and hasattr(args, 'dry_run') else settings.get("dry_run", False)
        min_size = args.min_size if args and hasattr(args, 'min_size') else settings.get("minimum_file_size", 0)
        ignore_older_than = args.ignore_older_than if args and hasattr(args, 'ignore_older_than') else settings.get("ignore_older_than", 0)
        move_files = args.move if args and hasattr(args, 'move') else settings.get("move_files", True)
        
        # Content analysis options
        use_content_analysis = args.content_analysis if args and hasattr(args, 'content_analysis') else False
        use_parallel = args.parallel if args and hasattr(args, 'parallel') else config.get("content_analysis", {}).get("processing", {}).get("parallel_processing", True)
        create_symlinks = args.symlinks if args and hasattr(args, 'symlinks') else config.get("content_analysis", {}).get("processing", {}).get("create_symlinks", True)
        base_dir = args.base_dir if args and hasattr(args, 'base_dir') and args.base_dir else directory
        
        # Build extension map from config
        extension_map = build_extension_map(config)
        
        directory = os.path.abspath(directory)
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist")
            return False
        
        # Log operation mode
        # Log operation mode
        if dry_run:
            logger.info(f"DRY RUN MODE: Starting organization preview of {directory}")
        else:
            logger.info(f"Starting organization of {directory}")
        
        # Log settings in use
        logger.info(f"Settings: min_size={min_size} bytes, ignore_older_than={ignore_older_than} days, {'move' if move_files else 'copy'} files")
        
        # Use content-based organization if enabled
        if use_content_analysis:
            if not CONTENT_ANALYSIS_AVAILABLE:
                logger.error("Content analysis requested but dependencies are not available")
                logger.info("Install required packages: pip install -r requirements.txt")
                return False
                
            logger.info(f"Using content-based organization with {'parallel' if use_parallel else 'sequential'} processing")
            logger.info(f"Multi-category files will be {'symlinked' if create_symlinks else 'copied'} to all matching categories")
            
            # Perform content-based organization
            try:
                # Initialize TextAnalyzer with config
                analyzer = TextAnalyzer(config_path=args.config if args and hasattr(args, 'config') else "config.yaml")
                
                # Set min_confidence if specified
                if args and hasattr(args, 'min_confidence'):
                    analyzer.min_confidence = args.min_confidence
                    logger.info(f"Set minimum confidence threshold to {args.min_confidence}")
                
                # Show keywords if requested
                show_keywords = args and hasattr(args, 'keywords') and args.keywords
                
                if dry_run:
                    logger.info("DRY RUN: Would perform content-based categorization")
                    # Just scan files and report what would happen
                    supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.md', '.rtf']
                    for file_path in Path(directory).glob('*'):
                        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                            try:
                                analysis = analyzer.analyze_file(str(file_path))
                                categories = analyzer.get_categories(analysis)
                                if categories:
                                    logger.info(f"Would categorize {file_path.name} into: {', '.join(categories)}")
                                    if show_keywords:
                                        keywords = analyzer.get_top_keywords(analysis, limit=5)
                                        if keywords:
                                            keyword_str = ", ".join([f"{k}:{round(s, 2)}" for k, s in keywords])
                                            logger.info(f"Top keywords: {keyword_str}")
                                else:
                                    logger.info(f"No categories found for {file_path.name}")
                            except Exception as e:
                                logger.error(f"Error analyzing {file_path.name}: {e}")
                else:
                    # Process files with content analysis
                    organize_files_by_content(directory, base_dir, analyzer, move_files, create_symlinks, use_parallel, min_size, ignore_older_than)
                
                return True
            except Exception as e:
                logger.error(f"Error during content-based organization: {e}")
                return False
        # First create all needed directories (skip in dry run mode)
        if not dry_run:
            create_directories(directory, extension_map)
        
        # Get all files in the directory (only at the top level, not in subfolders)
        # Get all files in the directory (only at the top level, not in subfolders)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        file_count = 0
        skipped_count = 0
        total_size = 0
        
        current_time = time.time()
        cutoff_time = current_time - (ignore_older_than * 86400)  # Convert days to seconds
        
        for filename in files:
            source_path = os.path.join(directory, filename)
            
            # Skip the script itself if it's in the same directory
            if os.path.abspath(source_path) == os.path.abspath(__file__):
                logger.debug(f"Skipping the script file: {filename}")
                continue
                
            # Skip configuration file
            if filename.lower() == "config.yaml":
                logger.debug(f"Skipping configuration file: {filename}")
                continue
                
            # Check file size if minimum size is specified
            file_size = os.path.getsize(source_path)
            if file_size < min_size:
                logger.info(f"Skipping {filename} - size ({human_readable_size(file_size)}) is below minimum threshold ({human_readable_size(min_size)})")
                skipped_count += 1
                continue
                
            # Check file modification time if ignore_older_than is specified
            if ignore_older_than > 0:
                mod_time = os.path.getmtime(source_path)
                if mod_time < cutoff_time:
                    logger.info(f"Skipping {filename} - modification date ({datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')}) is older than {ignore_older_than} days")
                    skipped_count += 1
                    continue
            # Get file extension (lowercase for case-insensitive matching)
            _, extension = os.path.splitext(filename.lower())
            
            # Determine target directory based on the extension map
            if extension in extension_map:
                target_dir = os.path.join(directory, extension_map[extension])
            else:
                target_dir = os.path.join(directory, "Other")
            
            # Create target path and ensure it's unique
            target_path = os.path.join(target_dir, filename)
            unique_target_path = get_unique_filename(target_path)
            
            # Log file details
            rel_path = os.path.relpath(unique_target_path, directory)
            file_info = f"{filename} ({human_readable_size(file_size)}) to {rel_path}"
            
            # In dry run mode, just log what would happen
            if dry_run:
                action = "Would move" if move_files else "Would copy"
                logger.info(f"{action}: {file_info}")
                file_count += 1
                total_size += file_size
                continue
            
            # Perform the actual file operation
            try:
                if move_files:
                    shutil.move(source_path, unique_target_path)
                    logger.info(f"Moved: {file_info}")
                else:
                    shutil.copy2(source_path, unique_target_path)
                    logger.info(f"Copied: {file_info}")
                    
                file_count += 1
                total_size += file_size
            except Exception as e:
                logger.error(f"Error {'moving' if move_files else 'copying'} {filename}: {e}")
        
        # Log summary
        action = "Would process" if dry_run else ("Moved" if move_files else "Copied")
        logger.info(f"Organization complete. {action} {file_count} files ({human_readable_size(total_size)}), skipped {skipped_count} files.")
        if dry_run:
            logger.info("This was a dry run. No files were actually modified.")
        elif not move_files:
            logger.info("Files were copied to destination folders. Original files remain in place.")
        return True
    except Exception as e:
        logger.error(f"Error during organization: {e}")
        return False
def organize_files_by_content(src_dir, dest_dir, analyzer, move_files=True, create_symlinks=True, parallel=True, min_size=0, ignore_older_than=0):
    """
    Organize files based on content analysis.
    
    Args:
        src_dir (str): Source directory containing files to organize
        dest_dir (str): Destination directory for organized files
        analyzer (TextAnalyzer): Initialized TextAnalyzer instance
        move_files (bool): Whether to move files (True) or copy them (False)
        create_symlinks (bool): Whether to create symlinks for multi-category files
        parallel (bool): Whether to use parallel processing
        min_size (int): Minimum file size in bytes to process
        ignore_older_than (int): Ignore files older than specified days
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    
    # Supported file types for text analysis
    supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.md', '.rtf']
    
    # Get list of files to process
    files = [f for f in src_dir.glob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
    logger.info(f"Found {len(files)} supported files for content analysis")
    
    # Apply filters
    if min_size > 0:
        files = [f for f in files if f.stat().st_size >= min_size]
        logger.info(f"{len(files)} files meet the minimum size requirement")
    
    if ignore_older_than > 0:
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=ignore_older_than)
        files = [f for f in files if datetime.datetime.fromtimestamp(f.stat().st_mtime) >= cutoff_time]
        logger.info(f"{len(files)} files meet the age requirement")
    
    # Prepare destination directories
    categories = list(analyzer.config.get('content_analysis', {}).get('tags', {}).keys())
    if categories:
        for category in categories:
            category_dir = dest_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created category directory: {category_dir}")
    
    # Create Unclassified folder
    unclassified_dir = dest_dir / "Unclassified"
    unclassified_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    processed_count = 0
    error_count = 0
    
    # Batch analyze if parallel is enabled
    if parallel and len(files) > 1:
        file_paths = [str(f) for f in files]
        results = analyzer.batch_analyze_files(file_paths, parallel=True)
        
        # Process results
        for file_path, analysis in results.items():
            file_path = Path(file_path)
            categories = analyzer.get_categories(analysis)
            _organize_single_file_by_content(file_path, dest_dir, categories, move_files, create_symlinks)
            processed_count += 1
    else:
        # Process files sequentially
        for file_path in files:
            try:
                analysis = analyzer.analyze_file(str(file_path))
                categories = analyzer.get_categories(analysis)
                _organize_single_file_by_content(file_path, dest_dir, categories, move_files, create_symlinks)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error analyzing {file_path.name}: {e}")
                error_count += 1
    
    logger.info(f"Content analysis organization complete. Processed {processed_count} files, {error_count} errors.")

def _organize_single_file_by_content(file_path, dest_dir, categories, move_files=True, create_symlinks=True):
    """
    Organize a single file based on its content categories.
    
    Args:
        file_path (Path): Path to the file
        dest_dir (Path): Base destination directory
        categories (List[str]): List of categories for the file
        move_files (bool): Whether to move or copy the file
        create_symlinks (bool): Whether to create symlinks for multi-category files
    """
    file_path = Path(file_path)
    dest_dir = Path(dest_dir)
    
    # If no categories found, put in Unclassified
    if not categories:
        target_dir = dest_dir / "Unclassified"
        target_path = target_dir / file_path.name
        target_path = Path(get_unique_filename(str(target_path)))
        
        try:
            if move_files:
                shutil.move(str(file_path), str(target_path))
                logger.info(f"Moved to Unclassified: {file_path.name}")
            else:
                shutil.copy2(str(file_path), str(target_path))
                logger.info(f"Copied to Unclassified: {file_path.name}")
        except Exception as e:
            logger.error(f"Error organizing {file_path.name}: {e}")
        
        return
    
    # Handle categorized files
    first_category = True
    for category in categories:
        target_dir = dest_dir / category
        target_path = target_dir / file_path.name
        target_path = Path(get_unique_filename(str(target_path)))
        
        try:
            if first_category:
                # Move or copy to first category
                if move_files:
                    shutil.move(str(file_path), str(target_path))
                    logger.info(f"Moved to {category}: {file_path.name}")
                else:
                    shutil.copy2(str(file_path), str(target_path))
                    logger.info(f"Copied to {category}: {file_path.name}")
                first_file_path = target_path
                first_category = False
            elif create_symlinks:
                # Create symlinks for additional categories
                target_path.symlink_to(first_file_path)
                logger.info(f"Created symlink in {category}: {file_path.name}")
            else:
                # Copy to all categories
                shutil.copy2(str(file_path), str(target_path))
                logger.info(f"Copied to {category}: {file_path.name}")
        except Exception as e:
            logger.error(f"Error organizing {file_path.name} to {category}: {e}")

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Organize files in a directory based on their extensions")
    parser.add_argument("directory", nargs="?", default=os.getcwd(),
                      help="Directory to organize (default: current directory)")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true",
                      help="Show what would be done without making any changes")
    parser.add_argument("--copy", dest="move", action="store_false",
                      help="Copy files instead of moving them")
    parser.add_argument("--min-size", type=int, default=0,
                      help="Minimum file size in bytes to process")
    parser.add_argument("--ignore-older-than", type=int, default=0,
                      help="Ignore files older than specified days")
    parser.add_argument("--config", default="config.yaml",
                      help="Path to configuration file (default: config.yaml)")
    
    # Content analysis related arguments
    content_group = parser.add_argument_group('Content Analysis', 'Options for content-based file organization')
    content_group.add_argument("--content-analysis", action="store_true",
                      help="Enable content-based categorization")
    content_group.add_argument("--parallel", action="store_true",
                      help="Enable parallel processing for content analysis")
    content_group.add_argument("--symlinks", action="store_true",
                      help="Create symlinks for multi-category files")
    content_group.add_argument("--base-dir", 
                      help="Specify output directory for categorized files")
    content_group.add_argument("--min-confidence", type=float, default=0.6,
                      help="Minimum confidence threshold for category assignment (0.0-1.0)")
    content_group.add_argument("--keywords", action="store_true",
                      help="Show detected keywords and their weights")
    
    return parser.parse_args()

def main():
    """
    Main function that processes the directory
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set logging level based on verbosity
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Get absolute path of target directory
        directory = os.path.abspath(args.directory)
        logger.info(f"Starting organization for directory: {directory}")
        
        # Load configuration from file
        config_path = args.config if hasattr(args, 'config') else "config.yaml"
        config = load_config(config_path)
        
        # Check content analysis dependencies if requested
        if args.content_analysis:
            if not check_content_analysis_dependencies():
                logger.warning("Continuing with extension-based organization only")
                args.content_analysis = False
        
        # Organize files and return appropriate exit code
        if organize_files(directory, config, args):
            logger.info("File organization completed successfully")
            return 0
        else:
            logger.error("File organization failed")
            return 1
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

