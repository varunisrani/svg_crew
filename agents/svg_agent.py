from utils.logging_config import setup_logging
from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv
import os
import time
from litellm import RateLimitError

load_dotenv()

logger = setup_logging()

class SVGAgent:
    def __init__(self):
        self.logger = setup_logging("svg_agent")
        self.logger.info("Initializing SVG Generator Agent...")
        try:
            llm = LLM(
                model="ft:gpt-4o-mini-2024-07-18:personal::ArguXr7z",
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.7,
                max_tokens=2000
            )
            self.logger.info("✓ LLM configuration successful for SVG Generator")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for SVG Generator: {str(e)}")
            raise
        
        self.logger.info("Setting up SVG Generator role and goals...")
        self.agent = Agent(
            role="SVG Code Generator",
            goal="Generate high-quality SVG code from design descriptions",
            backstory="""You are an SVG code generator assistant. Your goal is to create SVG code based on design descriptions provided by users. Your SVG code should accurately represent the design elements and layout described in the input.""",
            llm=llm,
            verbose=True
        )
        self.logger.info("✓ SVG Generator Agent successfully initialized with role and goals")

    def generate_svg(self, description):
        """Generate SVG code based on the description"""
        self.logger.info("Starting SVG generation...")
        self.logger.info(f"Input description: {description}")
        try:
            task = Task(
                description=f"""
                Generate an SVG testimonial design based on this description:
                {description}

                Follow these guidelines:
                1. Canvas size must be 1080x1080px
                2. Use clean typography and hierarchical text
                3. Layout must be centered
                4. Design must be minimal and professional
                5. SVG code must be valid and commented
                """,
                expected_output="Complete SVG code for the testimonial design",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            self.logger.info("Executing SVG generation task...")
            result = crew.kickoff()
            result_str = str(result.raw_output if hasattr(result, 'raw_output') else result)
            self.logger.info("✓ SVG generation completed successfully")
            self.logger.info(f"Generated SVG preview: {result_str[:200]}...")
            return result
        except Exception as e:
            self.logger.error(f"Error generating SVG: {str(e)}")
            raise

    def optimize_svg(self, svg_code):
        """Optimize the SVG code for performance and compatibility"""
        self.logger.info("Starting SVG optimization...")
        try:
            task = Task(
                description=f"""
                Optimize this SVG code for performance and readability:
                {svg_code}

                Follow these guidelines:
                1. Remove unnecessary attributes
                2. Optimize path data
                3. Combine similar styles
                4. Group related elements
                5. Add ARIA labels for accessibility
                """,
                expected_output="Optimized SVG code with improved performance and accessibility",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            self.logger.info("Executing SVG optimization task...")
            result = crew.kickoff()
            self.logger.info("✓ SVG optimization completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error optimizing SVG: {str(e)}")
            raise
