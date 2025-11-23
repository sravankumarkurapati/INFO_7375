"""
Test web scraper
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.web_scraper import web_scraper

def test_scraper():
    """Test web scraping functionality"""
    
    print("\n🌐 Testing Web Scraper...")
    print("=" * 60)
    
    # Test URLs
    test_urls = [
        "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need
        "https://arxiv.org/abs/1810.04805",  # BERT
    ]
    
    for idx, url in enumerate(test_urls, 1):
        print(f"\n{idx}. Testing URL: {url}")
        print("-" * 60)
        
        # Fetch content
        content = web_scraper.fetch_content(url)
        
        if content and content.get('success'):
            print(f"✅ Content fetched successfully")
            print(f"   Title: {content['title'][:60]}...")
            print(f"   Length: {content['length']} characters")
            print(f"   Metadata keys: {list(content['metadata'].keys())[:5]}")
            
            # Extract abstract
            abstract = web_scraper.extract_abstract(url)
            if abstract:
                print(f"\n   Abstract preview:")
                print(f"   {abstract[:200]}...")
        else:
            print(f"❌ Failed to fetch content")
    
    print("\n" + "=" * 60)
    print("✅ WEB SCRAPER TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_scraper()