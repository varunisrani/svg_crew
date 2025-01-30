from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class DescriptionAgent:
    def __init__(self):
        logger.info("Initializing Description Generator Agent...")
        try:
            llm = LLM(
                model="gpt-4o-mini",  # Changed from fine-tuned model to gpt-4o-mini
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.7,
                max_tokens=2000
            )
            logger.info("✓ LLM configuration successful for Description Generator")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for Description Generator: {str(e)}")
            raise
        
        logger.info("Setting up Description Generator role and goals...")
        self.agent = Agent(
            role="SVG Description Generator",
            goal="Generate detailed SVG descriptions from user requirements",
            backstory="""You are an expert in converting user requirements into 
            detailed SVG design descriptions, with a focus on visual aesthetics 
            and user experience.""",
            llm=llm,
            verbose=True
        )
        logger.info("✓ Description Generator Agent successfully initialized with role and goals")

    def generate_description(self, requirements):
        """Generate a detailed SVG description based on requirements"""
        logger.info("Starting description generation...")
        try:
            task = Task(
                description=f"""
                Create a detailed SVG testimonial description based on these requirements:
                {requirements}
                """,
                expected_output="A detailed SVG design description including layout, colors, typography, and visual elements",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing description generation task...")
            result = crew.kickoff()
            logger.info("✓ Description generation completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            raise

    def refine_description(self, description, feedback):
        """Refine the description based on feedback"""
        logger.info("Starting description refinement...")
        try:
            task = Task(
                description=f"""
                Refine the following SVG description based on feedback:
                
                Original Description:
                {description}
                
                Feedback:
                {feedback}
                """,
                expected_output="A refined SVG description incorporating the provided feedback",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing description refinement task...")
            result = crew.kickoff()
            logger.info("✓ Description refinement completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error refining description: {str(e)}")
            raise
