#!/usr/bin/env python3
import os
import shutil
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('file_organizer')

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

def create_directories(base_dir):
    """
    Create the necessary directories based on the extension map
    
    Args:
        base_dir (str): Base directory where folders will be created
    """
    try:
        # Create a set of unique directory paths from the extension map
        unique_dirs = set(EXTENSION_MAP.values())
        
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

def organize_files(directory):
    """
    Organize files in the specified directory based on their extensions
    
    Args:
        directory (str): Directory to organize
    """
    try:
        directory = os.path.abspath(directory)
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist")
            return False
        
        logger.info(f"Starting organization of {directory}")
        
        # First create all needed directories
        create_directories(directory)
        
        # Get all files in the directory (only at the top level, not in subfolders)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        file_count = 0
        for filename in files:
            source_path = os.path.join(directory, filename)
            
            # Skip the script itself if it's in the same directory
            if os.path.samefile(source_path, __file__):
                logger.debug(f"Skipping the script file: {filename}")
                continue
            
            # Get file extension (lowercase for case-insensitive matching)
            _, extension = os.path.splitext(filename.lower())
            
            # Determine target directory
            if extension in EXTENSION_MAP:
                target_dir = os.path.join(directory, EXTENSION_MAP[extension])
            else:
                target_dir = os.path.join(directory, "Other")
            
            # Create target path and ensure it's unique
            target_path = os.path.join(target_dir, filename)
            unique_target_path = get_unique_filename(target_path)
            
            # Move the file
            try:
                shutil.move(source_path, unique_target_path)
                logger.info(f"Moved: {filename} to {os.path.relpath(unique_target_path, directory)}")
                file_count += 1
            except Exception as e:
                logger.error(f"Error moving {filename}: {e}")
        
        logger.info(f"Organization complete. Moved {file_count} files.")
        return True
    except Exception as e:
        logger.error(f"Error during organization: {e}")
        return False

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
    
    return parser.parse_args()

def main():
    """Main function that processes the directory"""
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    directory = os.path.abspath(args.directory)
    logger.info(f"Starting organization for directory: {directory}")
    
    if organize_files(directory):
        logger.info("File organization completed successfully")
    else:
        logger.error("File organization failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

