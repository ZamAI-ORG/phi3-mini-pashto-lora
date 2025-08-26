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
        # Default headers to appear as a regular browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        # Most requests libraries handle Accept-Encoding automatically; avoid forcing it here

        # Track last fetched page URL so we can set Referer for subsequent PDF downloads
        self.last_page_url: Optional[str] = None
        
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
            logger.info("Successfully parsed page content")
            # remember this page as the referer for subsequent downloads
            self.last_page_url = url
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
    
    def _fetch_response(self, url: str, headers: dict, timeout: int) -> Optional[requests.Response]:
        """Fetch a URL and return the Response or None on error (logs errors)."""
        try:
            resp = self.session.get(url, headers=headers, stream=True, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            logger.error("Error fetching page %s: %s", url, exc)
            return None

    def _extract_filename_from_cd(self, cd: Optional[str]) -> Optional[str]:
        """Extract filename from Content-Disposition header if present."""
        if not cd:
            return None
        import urllib.parse as _urlparse
        m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^;"\']+)', cd)
        return _urlparse.unquote(m.group(1)) if m else None

    def _filename_from_url(self, u: str) -> str:
        """Derive a filename from the URL path."""
        import urllib.parse as _urlparse
        path = _urlparse.urlparse(u).path or ""
        base = os.path.basename(path)
        return base or "download.pdf"

    def _resolve_filename(self, resp: requests.Response, url: str, provided: Optional[str]) -> str:
        """Decide the filename to use, sanitize it and return."""
        cd = resp.headers.get("content-disposition")
        candidate = provided or self._extract_filename_from_cd(cd) or self._filename_from_url(url)
        return self.sanitize_filename(candidate)

    def _save_stream_to_file(self, resp: requests.Response, out_path: str) -> bool:
        """Stream write response content to out_path, remove partial file on failure."""
        try:
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    fh.write(chunk)
            return True
        except Exception as exc:
            logger.error("Failed to save PDF %s -> %s: %s", resp.url if hasattr(resp, "url") else "<url>", out_path, exc)
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return False

    def download_pdf(self, url: str, filename: Optional[str] = None) -> bool:
        """
        Download a single PDF file. Returns True on success, False on failure.
        Refactored to use small helpers for readability and lower cognitive complexity.
        """
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PDFDownloader/1.0)"}
        timeout = 15

        # Fetch resource and return early on failure
        resp = self._fetch_response(url, headers, timeout)
        if resp is None:
            return False

        # Resolve filename and target path
        candidate = self._resolve_filename(resp, url, filename)
        out_dir = getattr(self, "output_dir", ".")
        out_path = os.path.join(out_dir, candidate)

        # Save stream to file (handles cleanup on failure)
        if not self._save_stream_to_file(resp, out_path):
            return False

        logger.info("Downloaded %s -> %s", url, out_path)
        return True
    
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

        logger.info("Download process completed!")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info(f"Files saved to: {self.output_dir.absolute()}")


def parse_args():
    """Parse and return CLI arguments."""
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

    parser.add_argument('--url', required=False,
                        help='URL of the webpage containing PDF links (use --url or provide as last positional argument)')
    parser.add_argument('pos_url', nargs='?', help='Positional URL (alternative to --url)')
    parser.add_argument('--output-dir', default='books', help='Directory to save downloaded PDFs (default: books)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between downloads in seconds (default: 1.0)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--user-agent', default=None, help='Custom User-Agent string to use (overrides default)')
    parser.add_argument('--dry-run', action='store_true', help='Only fetch the page and list found PDF links without downloading')

    return parser.parse_args()


def validate_url(url: str) -> None:
    """Validate the provided URL or exit the program with error."""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")
    except Exception as e:
        logger.error(f"Invalid URL: {url} - {e}")
        sys.exit(1)


def run_dry_run(downloader: PDFDownloader, url: str) -> None:
    """Perform dry-run: fetch page and print found PDF links, then exit."""
    soup = downloader.get_page_content(url)
    if soup is None:
        logger.error("Failed to fetch webpage content. Exiting.")
        sys.exit(1)
    pdf_links = downloader.find_pdf_links(soup, url)
    if not pdf_links:
        logger.info("No PDF links found on the webpage.")
    else:
        logger.info(f"Found {len(pdf_links)} PDF links (dry-run). Listing up to 50:")
        for i, link in enumerate(pdf_links[:50], 1):
            print(f"{i}. {link}")
    sys.exit(0)


def run_download(downloader: PDFDownloader, url: str) -> None:
    """Run the actual download process."""
    downloader.download_all_pdfs(url)


def main():
    """Main function to run the PDF downloader (refactored)."""
    args = parse_args()

    # Set log level early if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If URL was provided positionally, use it as a fallback for --url
    if not getattr(args, 'url', None) and getattr(args, 'pos_url', None):
        args.url = args.pos_url

    # Validate URL
    if not getattr(args, 'url', None):
        logger.error("No URL provided. Use --url or provide the URL as the last positional argument.")
        sys.exit(1)
    validate_url(args.url)

    # Create downloader and apply options
    downloader = PDFDownloader(output_dir=args.output_dir, delay=args.delay)
    if args.user_agent:
        downloader.session.headers['User-Agent'] = args.user_agent

    # Execute chosen mode with minimal control flow in main
    try:
        if args.dry_run:
            run_dry_run(downloader, args.url)
        else:
            run_download(downloader, args.url)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()