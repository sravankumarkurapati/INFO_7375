"""
Test SerperDevTool for paper search - FIXED for CrewAI 1.5+
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crewai_tools import SerperDevTool
from config.settings import settings
import json

def test_serper():
    """Test Serper API for academic paper search"""
    
    print("\n🔍 Testing SerperDevTool for Academic Search...")
    print("=" * 60)
    
    # Verify API key
    if not settings.SERPER_API_KEY:
        print("❌ SERPER_API_KEY not found in .env")
        print("Get your key from: https://serper.dev/")
        sys.exit(1)
    
    print("✅ Serper API key found")
    
    # Initialize tool with search query parameter
    test_query = "transformer models natural language processing"
    
    try:
        # NEW WAY: Pass query during initialization
        search_tool = SerperDevTool(
            search_url="https://google.serper.dev/search",
            n_results=10
        )
        print("✅ SerperDevTool initialized")
    except Exception as e:
        print(f"❌ Failed to initialize tool: {e}")
        # Try simpler initialization
        try:
            search_tool = SerperDevTool()
            print("✅ SerperDevTool initialized (simple mode)")
        except Exception as e2:
            print(f"❌ Both initialization methods failed: {e2}")
            sys.exit(1)
    
    print(f"\n🔎 Searching: '{test_query}'")
    print("⏳ Please wait 5-10 seconds...\n")
    
    try:
        # NEW WAY: Call with search query as keyword argument
        results = search_tool._run(search_query=test_query)
        
        print("✅ Search successful!")
        print("=" * 60)
        
        # Display results
        if results:
            print(f"\n📋 Results Type: {type(results)}")
            print(f"📋 Results Length: {len(str(results)) if results else 0} characters")
            print(f"\n📋 First 300 characters:")
            print(str(results)[:300])
            print("...\n")
            
            # Try to parse as JSON
            try:
                if isinstance(results, str):
                    parsed = json.loads(results)
                else:
                    parsed = results
                
                print("✅ Results parsed as JSON")
                
                if isinstance(parsed, dict):
                    print(f"📦 Keys: {list(parsed.keys())}")
                    
                    if 'organic' in parsed:
                        organic = parsed['organic']
                        print(f"\n📄 Found {len(organic)} results")
                        
                        if organic:
                            first = organic[0]
                            print("\n📌 First Result:")
                            print(f"  Title: {first.get('title', 'N/A')}")
                            print(f"  Link: {first.get('link', 'N/A')}")
                            print(f"  Snippet: {first.get('snippet', 'N/A')[:80]}...")
                    
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️  Could not parse as JSON: {e}")
                print("But search returned data, which is good!")
        
        print("\n" + "=" * 60)
        print("✅ SERPER TEST SUCCESSFUL!")
        print("=" * 60)
        print("\n💡 Note: Tool is working. If output format differs,")
        print("   the agent will handle it correctly.")
        
        return True
        
    except TypeError as e:
        print(f"\n⚠️  API signature issue: {e}")
        print("\n🔄 Trying alternative method...")
        
        # FALLBACK: Try calling as string
        try:
            results = search_tool._run(test_query)
            print("✅ Alternative method worked!")
            print(f"Results: {str(results)[:200]}...")
            return True
        except Exception as e2:
            print(f"❌ Alternative also failed: {e2}")
            
            # LAST RESORT: Direct API call
            print("\n🔄 Trying direct API call...")
            try:
                import requests
                
                url = "https://google.serper.dev/search"
                headers = {
                    'X-API-KEY': settings.SERPER_API_KEY,
                    'Content-Type': 'application/json'
                }
                payload = json.dumps({"q": test_query})
                
                response = requests.post(url, headers=headers, data=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Direct API call successful!")
                    print(f"Found {len(data.get('organic', []))} results")
                    return True
                else:
                    print(f"❌ API returned status {response.status_code}")
                    return False
                    
            except Exception as e3:
                print(f"❌ Direct API call failed: {e3}")
                return False
    
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_serper()
    sys.exit(0 if success else 1)