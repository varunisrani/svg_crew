from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv
import os
import logging
import time
from litellm import RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

class SVGAgent:
    def __init__(self):
        logger.info("Initializing SVG Generator Agent...")
        try:
            llm = LLM(
                model="gpt-4o-mini",
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.7,
                max_tokens=2000
            )
            logger.info("✓ LLM configuration successful for SVG Generator")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for SVG Generator: {str(e)}")
            raise
        
        logger.info("Setting up SVG Generator role and goals...")
        self.agent = Agent(
            role="SVG Code Generator",
            goal="Generate high-quality SVG code from design descriptions",
            backstory="""You are an expert SVG developer specialized in 
            creating beautiful and efficient SVG testimonials from detailed 
            design descriptions.""",
            llm=llm,
            verbose=True
        )
        logger.info("✓ SVG Generator Agent successfully initialized with role and goals")

        self.color_palettes = {
            'beige_minimalist': {'bg': '#E8E1D9', 'accent': '#9B4F3E'},
            'teal_modern': {'bg': '#65A7A1', 'container': '#F5EBE4'},
            'white_pink': {'bg': '#FFFFFF', 'circle': '#F3E5F5'},
            'monochrome': {'bg': '#B0B0B0', 'container': '#FFFFFF'},
            'yellow_white': {'bg': '#E7B831', 'container': '#FFFFFF'}
        }

    def generate_svg(self, description):
        """Generate SVG code based on the description"""
        logger.info("Starting SVG code generation...")
        try:
            task = Task(
                description=f"""
                Generate SVG code for testimonial based on this description:
                {description}
                """,
                expected_output="Complete SVG code for the testimonial design",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing SVG generation task...")
            result = crew.kickoff()
            logger.info("✓ SVG generation completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating SVG: {str(e)}")
            raise

    def optimize_svg(self, svg_code):
        """Optimize the SVG code for performance and compatibility"""
        logger.info("Starting SVG optimization...")
        try:
            task = Task(
                description=f"""
                Optimize the following SVG code:
                {svg_code}
                
                Perform these optimizations:
                1. Remove unnecessary attributes and elements
                2. Optimize path data
                3. Combine similar styles
                4. Ensure proper grouping
                5. Add ARIA labels for accessibility
                """,
                expected_output="Optimized SVG code with improved performance and accessibility",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing SVG optimization task...")
            result = crew.kickoff()
            logger.info("✓ SVG optimization completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error optimizing SVG: {str(e)}")
            raise
