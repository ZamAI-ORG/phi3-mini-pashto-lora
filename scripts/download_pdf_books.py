#!/usr/bin/env python3
"""
PDF Book Downloader Script

This script downloads all PDF books from a specified webpage.
It's specifically designed for the Afghan Ministry of Education website
but can be adapted for other educational resources.

Usage:
    python scripts/download_pdf_books.py --url "https://moe.gov.af/..." --output-dir "books/"

Legal Notice:
    This script is intended for educational and research purposes only.
    Please ensure you have permission to download content from the target website
    and comply with the website's terms of service and copyright laws.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class PDFDownloader:
    """A class to download PDF files from a webpage."""
    
    def __init__(self, output_dir: str = "books", delay: float = 1.0):
        """
        Initialize the PDF downloader.
        
        Args:
            output_dir: Directory to save downloaded PDFs
            delay: Delay between requests to be respectful to the server
        """
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        
        # Set a user agent to identify our scraper
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational PDF Downloader for Research Purposes)'
        })
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized PDF downloader with output directory: {self.output_dir}")
    
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse the content of a webpage.
        
        Args:
            url: The URL to fetch
            
        Returns:
            BeautifulSoup object of the parsed page or None if failed
        """
        try:
            logger.info(f"Fetching page content from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to detect encoding
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.content, 'html.parser')
            logger.info(f"Successfully parsed page content")
            return soup
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing page {url}: {e}")
            return None
    
    def find_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find all PDF links on a webpage.
        
        Args:
            soup: BeautifulSoup object of the parsed page
            base_url: Base URL for resolving relative links
            
        Returns:
            List of PDF URLs
        """
        pdf_links = []
        
        # Find all links that could be PDFs
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Check if the link points to a PDF file
            if href.lower().endswith('.pdf') or 'pdf' in href.lower():
                full_url = urljoin(base_url, href)
                pdf_links.append(full_url)
                logger.debug(f"Found PDF link: {full_url}")
        
        # Also check for links in onclick or data attributes that might contain PDF URLs
        for element in soup.find_all(attrs={'onclick': True}):
            onclick = element['onclick']
            pdf_matches = re.findall(r'["\']([^"\']*\.pdf[^"\']*)["\']', onclick, re.IGNORECASE)
            for match in pdf_matches:
                full_url = urljoin(base_url, match)
                pdf_links.append(full_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in pdf_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        logger.info(f"Found {len(unique_links)} unique PDF links")
        return unique_links
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename for safe filesystem usage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename.strip('._')
        
        # Ensure it's not too long (limit to 255 characters)
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext
        
        return filename
    
    def download_pdf(self, url: str, filename: Optional[str] = None) -> bool:
        """
        Download a single PDF file.
        
        Args:
            url: URL of the PDF to download
            filename: Optional custom filename
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Generate filename if not provided
            if filename is None:
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename or not filename.lower().endswith('.pdf'):
                    filename = f"book_{hash(url) % 10000}.pdf"
            
            filename = self.sanitize_filename(filename)
            filepath = self.output_dir / filename
            
            # Skip if file already exists
            if filepath.exists():
                logger.info(f"File already exists, skipping: {filename}")
                return True
            
            logger.info(f"Downloading: {url} -> {filename}")
            
            # Download with streaming to handle large files
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Check if the response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not filename.lower().endswith('.pdf'):
                logger.warning(f"URL may not be a PDF file: {url} (Content-Type: {content_type})")
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"Successfully downloaded: {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    
    def download_all_pdfs(self, url: str) -> None:
        """
        Download all PDF files from a webpage.
        
        Args:
            url: URL of the webpage to scrape
        """
        logger.info(f"Starting PDF download process for: {url}")
        
        # Get the webpage content
        soup = self.get_page_content(url)
        if soup is None:
            logger.error("Failed to fetch webpage content. Exiting.")
            return
        
        # Find all PDF links
        pdf_links = self.find_pdf_links(soup, url)
        
        if not pdf_links:
            logger.warning("No PDF links found on the webpage.")
            return
        
        # Download each PDF
        successful_downloads = 0
        failed_downloads = 0
        
        for i, pdf_url in enumerate(pdf_links, 1):
            logger.info(f"Processing PDF {i}/{len(pdf_links)}")
            
            if self.download_pdf(pdf_url):
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Be respectful to the server
            if i < len(pdf_links):  # Don't delay after the last download
                time.sleep(self.delay)
        
        logger.info(f"Download process completed!")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info(f"Files saved to: {self.output_dir.absolute()}")


def main():
    """Main function to run the PDF downloader."""
    parser = argparse.ArgumentParser(
        description="Download PDF books from a webpage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_pdf_books.py --url "https://moe.gov.af/..." --output-dir "books/"
  python scripts/download_pdf_books.py --url "https://example.com/books" --delay 2.0

Legal Notice:
  This script is for educational and research purposes only.
  Ensure you have permission to download content and comply with
  the website's terms of service and copyright laws.
        """
    )
    
    parser.add_argument(
        '--url',
        required=True,
        help='URL of the webpage containing PDF links'
    )
    
    parser.add_argument(
        '--output-dir',
        default='books',
        help='Directory to save downloaded PDFs (default: books)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between downloads in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate URL
    try:
        parsed = urlparse(args.url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")
    except Exception as e:
        logger.error(f"Invalid URL: {args.url} - {e}")
        sys.exit(1)
    
    # Create downloader and start downloading
    downloader = PDFDownloader(output_dir=args.output_dir, delay=args.delay)
    
    try:
        downloader.download_all_pdfs(args.url)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()