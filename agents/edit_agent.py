from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv
import os
import logging
import time
from litellm import RateLimitError

logger = logging.getLogger(__name__)

load_dotenv()

class EditAgent:
    def __init__(self, llm=None):
        logger.info("Initializing SVG Editor Agent...")
        
        if llm is None:
            max_retries = 5
            retry_delay = 10
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = retry_delay * (2 ** (attempt - 1))
                        logger.warning(f"Waiting {wait_time} seconds before attempt {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    
                    llm = LLM(
                        model="gpt-4o-mini",
                        api_key=os.getenv('OPENAI_API_KEY'),
                        temperature=0.7,
                        max_tokens=2000
                    )
                    logger.info("✓ LLM configuration successful for Editor Agent")
                    break
                except RateLimitError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed all retries for Editor Agent: {str(e)}")
                        raise
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM for Editor Agent: {str(e)}")
                    raise
        
        logger.info("Setting up Editor Agent role and goals...")
        self.agent = Agent(
            role="SVG Code Editor",
            goal="Review and optimize SVG code for best practices and performance",
            backstory="""You are an expert SVG code editor with deep knowledge 
            of SVG optimization techniques and best practices.""",
            llm=llm,
            verbose=True
        )
        logger.info("✓ Editor Agent successfully initialized with role and goals")

    def analyze_svg(self, svg_code, rendered_image=None):
        """Analyze SVG code and rendered image for issues"""
        logger.info("Starting SVG code analysis...")
        try:
            task = Task(
                description=f"""
                Analyze this SVG code for issues:
                {svg_code}
                """,
                expected_output="Detailed analysis of SVG code issues and recommendations",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing SVG analysis task...")
            result = crew.kickoff()
            logger.info("✓ SVG analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error analyzing SVG: {str(e)}")
            raise

    def fix_svg(self, svg_code, issues):
        """Fix identified issues in the SVG code"""
        logger.info("Starting SVG code fixes...")
        try:
            task = Task(
                description=f"""
                Fix these issues in the SVG code:
                
                SVG Code:
                {svg_code}
                
                Issues to Fix:
                {issues}
                """,
                expected_output="Fixed SVG code with all issues resolved",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing SVG fix task...")
            result = crew.kickoff()
            logger.info("✓ SVG fixes completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error fixing SVG: {str(e)}")
            raise
