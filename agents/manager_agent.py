from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
import logging
import time
from litellm import RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

class ManagerAgent:
    def __init__(self, llm=None):
        logger.info("Initializing Manager Agent...")
        
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
                    logger.info("✓ LLM configuration successful for Manager Agent")
                    break
                except RateLimitError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed all retries for Manager Agent: {str(e)}")
                        raise
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM for Manager Agent: {str(e)}")
                    raise
        
        logger.info("Setting up Manager Agent role and goals...")
        self.agent = Agent(
            role="SVG Project Manager",
            goal="Manage and coordinate the SVG testimonial generation process",
            backstory="""You are an expert project manager specialized in 
            coordinating SVG testimonial generation, ensuring high quality 
            and consistency.""",
            llm=llm,
            verbose=True
        )
        logger.info("✓ Manager Agent successfully initialized with role and goals")

    def analyze_request(self, request):
        """Analyze user request and coordinate with other agents"""
        task = Task(
            description=f"""
            Analyze the following testimonial request and create a structured plan:
            {request}
            
            1. Extract key requirements
            2. Identify design elements needed
            3. Create a task list for other agents
            4. Ensure all requirements are covered
            """,
            expected_output="A structured analysis of the testimonial request including requirements, design elements, and task list",
            agent=self.agent
        )
        
        # Create a crew with just this task
        crew = Crew(
            agents=[self.agent],
            tasks=[task]
        )
        
        # Execute the task through the crew
        result = crew.kickoff()
        return result

    def review_svg(self, svg_code):
        """Review generated SVG and provide feedback"""
        task = Task(
            description=f"""
            Review the following SVG code and identify any issues:
            {svg_code}
            
            Provide a detailed analysis focusing on:
            1. Visual consistency
            2. Code structure
            3. Accessibility
            4. Performance
            """,
            expected_output="A detailed analysis of the SVG code with identified issues and recommendations",
            agent=self.agent
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task]
        )
        
        result = crew.kickoff()
        return result

    def some_method(self):
        response = self.agent.run("Your prompt here")
        return response
