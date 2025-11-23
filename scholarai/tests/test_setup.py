"""
Test basic CrewAI setup - FIXED with proper paths
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from config.settings import settings

def test_basic_agent():
    """Test basic agent creation and execution"""
    
    print("🧪 Testing Basic CrewAI Setup...")
    print("=" * 50)
    
    # Validate settings
    try:
        settings.validate()
        print("✅ Settings validated")
    except Exception as e:
        print(f"❌ Settings validation failed: {e}")
        print("\n💡 Make sure you:")
        print("  1. Created .env file with API keys")
        print("  2. Have valid OpenAI and Serper API keys")
        sys.exit(1)
    
    # Create LLM
    try:
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )
        print("✅ LLM initialized")
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        sys.exit(1)
    
    # Create test agent
    try:
        test_agent = Agent(
            role='Test Agent',
            goal='Verify that CrewAI is working correctly',
            backstory='You are a test agent designed to validate system setup.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        print("✅ Agent created")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        sys.exit(1)
    
    # Create test task
    try:
        test_task = Task(
            description='Say "Hello from ScholarAI! System is working." and list 3 benefits of using AI for research.',
            expected_output='A greeting message and a list of 3 benefits',
            agent=test_agent
        )
        print("✅ Task created")
    except Exception as e:
        print(f"❌ Task creation failed: {e}")
        sys.exit(1)
    
    # Create and run crew
    try:
        crew = Crew(
            agents=[test_agent],
            tasks=[test_task],
            process=Process.sequential,
            verbose=True
        )
        print("✅ Crew created")
        print("\n🚀 Executing test task...\n")
        print("⏳ This may take 10-30 seconds...\n")
        
        result = crew.kickoff()
        
        print("\n" + "=" * 50)
        print("✅ TEST SUCCESSFUL!")
        print("=" * 50)
        print("\n📋 Result:")
        print(result)
        print("\n" + "=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Crew execution failed: {e}")
        print("\nℹ️  This might be due to:")
        print("  1. Invalid API key")
        print("  2. No OpenAI credits")
        print("  3. Network issue")
        print("\nTrying with simpler configuration...")
        
        # Fallback: Try without verbose
        try:
            crew = Crew(
                agents=[test_agent],
                tasks=[test_task],
                process=Process.sequential,
                verbose=False
            )
            result = crew.kickoff()
            print("\n✅ TEST SUCCESSFUL (with simplified config)!")
            print(f"Result: {result}")
            return True
        except Exception as e2:
            print(f"\n❌ Fallback also failed: {e2}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_basic_agent()
    sys.exit(0 if success else 1)