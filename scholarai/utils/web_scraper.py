"""
Web scraping utilities for content extraction
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import time
from utils.logger import logger

class WebScraper:
    """Web content scraper with retry logic"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize web scraper
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_content(self, url: str) -> Optional[Dict]:
        """
        Fetch and parse web content
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with content or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching content from: {url[:60]}...")
                
                response = self.session.get(
                    url, 
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    content = self._parse_content(response.text, url)
                    logger.info(f"✅ Content fetched successfully ({len(content.get('text', ''))} chars)")
                    return content
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.RequestException as e:
                logger.warning(f"Request failed: {e}")
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        logger.error(f"❌ Failed to fetch content after {self.max_retries} attempts")
        return None
    
    def _parse_content(self, html: str, url: str) -> Dict:
        """
        Parse HTML content
        
        Args:
            html: HTML string
            url: Source URL
            
        Returns:
            Parsed content dictionary
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        # Extract main text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        return {
            'url': url,
            'title': title,
            'text': text,
            'metadata': metadata,
            'length': len(text),
            'success': True
        }
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from HTML"""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        return metadata
    
    def extract_abstract(self, url: str) -> Optional[str]:
        """
        Extract abstract/summary from paper URL
        
        Args:
            url: Paper URL
            
        Returns:
            Abstract text or None
        """
        content = self.fetch_content(url)
        if not content:
            return None
        
        text = content['text']
        
        # Look for abstract section
        text_lower = text.lower()
        
        # Try to find abstract
        abstract_markers = ['abstract', 'summary']
        for marker in abstract_markers:
            if marker in text_lower:
                # Find the section
                start_idx = text_lower.find(marker)
                # Take next 500 characters as abstract
                abstract = text[start_idx:start_idx + 1000]
                
                # Clean up
                lines = abstract.split('\n')
                abstract = ' '.join(lines[1:6])  # Skip title, take next 5 lines
                
                if len(abstract) > 50:  # Valid abstract
                    return abstract.strip()
        
        # Fallback: return first 500 chars
        return text[:500].strip()

# Singleton instance
web_scraper = WebScraper()