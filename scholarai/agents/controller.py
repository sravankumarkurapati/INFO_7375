"""
Controller Agent - Orchestrates the entire research workflow
"""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import Dict, List
from config.settings import settings
from utils.logger import logger

class ControllerAgent:
    """
    Main controller agent that orchestrates all other agents
    """
    
    def __init__(self):
        """Initialize Controller Agent"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.agent = Agent(
            role='Research Project Manager',
            goal='Orchestrate the research workflow by delegating tasks to specialized agents and ensuring high-quality output',
            backstory="""You are an experienced research project manager who coordinates 
            complex academic research projects. You excel at:
            - Breaking down research queries into manageable tasks
            - Delegating tasks to the right specialists
            - Validating quality at each stage
            - Making decisions about when to proceed or refine
            - Ensuring coherent final outputs
            
            You manage a team of specialist agents:
            1. Paper Hunter - Finds relevant academic papers
            2. Content Analyzer - Analyzes paper content
            3. Research Synthesizer - Identifies gaps and trends
            4. Quality Reviewer - Validates output quality
            
            You coordinate their work to produce comprehensive research analyses.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        logger.info("Controller Agent initialized")
    
    def create_workflow(
        self,
        query: str,
        paper_hunter: Agent,
        content_analyzer: Agent,
        synthesizer: Agent,
        quality_reviewer: Agent
    ) -> Crew:
        """
        Create orchestrated CrewAI workflow
        
        Args:
            query: Research query
            All specialist agents
            
        Returns:
            Configured Crew object
        """
        # Task 1: Paper Discovery
        task1 = Task(
            description=f"""Search for academic papers on: "{query}"
            
            Use SerperDevTool to find 10-15 relevant papers.
            Return a structured list with title, URL, year, and source.""",
            expected_output="List of 10-15 papers with metadata",
            agent=paper_hunter
        )
        
        # Task 2: Content Analysis
        task2 = Task(
            description=f"""Analyze the papers from the previous task.
            
            For each paper, extract:
            - Key findings
            - Research methodology
            - Main contributions
            - Technical terms
            
            Use web scraping tools as needed.""",
            expected_output="Detailed analysis of each paper",
            agent=content_analyzer,
            context=[task1]
        )
        
        # Task 3: Research Synthesis
        task3 = Task(
            description=f"""Synthesize research using the custom Research Gap Analyzer.
            
            Identify:
            - Research gaps
            - Trends
            - Recommendations
            
            Create visualizations.""",
            expected_output="Comprehensive synthesis with gaps and recommendations",
            agent=synthesizer,
            context=[task1, task2]
        )
        
        # Task 4: Quality Review
        task4 = Task(
            description=f"""Review the overall research quality.
            
            Evaluate completeness, evidence quality, and coherence.
            Provide scores and identify any needed refinements.""",
            expected_output="Quality assessment with scores and recommendations",
            agent=quality_reviewer,
            context=[task1, task2, task3]
        )
        
        # Create Crew with sequential process
        crew = Crew(
            agents=[paper_hunter, content_analyzer, synthesizer, quality_reviewer],
            tasks=[task1, task2, task3, task4],
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("Workflow crew created with 4 tasks")
        return crew
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent"""
        return self.agent

def create_controller() -> ControllerAgent:
    """Create and return Controller agent"""
    return ControllerAgent()