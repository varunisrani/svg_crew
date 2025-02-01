from utils.logging_config import setup_logging
from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv
import os

load_dotenv()

class DescriptionAgent:
    def __init__(self):
        self.logger = setup_logging("description_agent")
        self.logger.info("Initializing Description Generator Agent...")
        try:
            llm = LLM(
                model="ft:gpt-4o-mini-2024-07-18:personal::AtrjXCvd",
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.9,  # Increased for more creativity
                max_tokens=2000
            )
            self.logger.info("✓ LLM configuration successful for Description Generator")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for Description Generator: {str(e)}")
            raise
        
        self.logger.info("Setting up Description Generator role and goals...")
        self.agent = Agent(
            role="SVG Testimonial Description Generator",
            goal="Create unique and detailed SVG testimonial design descriptions",
            backstory="""You are a highly creative SVG Testimonial Description Generator. 
            Your role is to create unique, detailed descriptions for SVG testimonials based on specific requirements.
            Each description must be original and tailored to the given specifications.""",
            llm=llm,
            verbose=True
        )
        self.logger.info("✓ Description Generator Agent successfully initialized with role and goals")

    def generate_description(self, requirements):
        """Generate a detailed SVG description based on requirements"""
        self.logger.info("Starting description generation...")
        self.logger.info(f"Input requirements: {requirements}")
        try:
            task = Task(
                description=f"""
                Create a UNIQUE and DETAILED SVG testimonial design description based on these requirements:
                {requirements}

                Your description MUST include:
                1. Background Design:
                   - Specific colors (use hex codes)
                   - Gradients or patterns if applicable
                   - Overall style and mood

                2. Layout Structure:
                   - Exact coordinates (x, y) for all elements
                   - Width and height specifications
                   - Alignment details

                3. Typography Details:
                   - Font families
                   - Sizes in pixels
                   - Colors (hex codes)
                   - Text alignment properties

                4. Container Specifications:
                   - Shape and dimensions
                   - Background colors
                   - Border properties
                   - Shadow effects if any

                5. Decorative Elements:
                   - Icons or graphics
                   - Positioning
                   - Colors and styles

                IMPORTANT:
                - DO NOT copy the example from the prompt
                - Generate a completely new and unique design
                - Use the specific requirements provided
                - Include precise measurements and coordinates
                - Ensure all colors match the requirements
                - Make the design match the target audience

                Format your response as a detailed description, including all technical specifications while maintaining readability.
                """,
                expected_output="A unique, detailed SVG design description with precise specifications",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            self.logger.info("Executing description generation task...")
            result = crew.kickoff()
            result_str = str(result.raw_output if hasattr(result, 'raw_output') else result)
            self.logger.info("✓ Description generation completed successfully")
            self.logger.info(f"Generated description: {result_str}")
            return result
        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            raise

    def refine_description(self, description, feedback):
        """Refine the description based on feedback"""
        self.logger.info("Starting description refinement...")
        try:
            task = Task(
                description=f"""
                Refine this SVG testimonial description based on the feedback:

                Original Description:
                {description}

                Feedback:
                {feedback}

                Guidelines for refinement:
                1. Address all feedback points
                2. Maintain technical precision
                3. Keep the design unique
                4. Ensure all measurements are accurate
                5. Preserve the original style while improving it
                """,
                expected_output="A refined SVG description incorporating the feedback",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            self.logger.info("Executing description refinement task...")
            result = crew.kickoff()
            self.logger.info("✓ Description refinement completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error refining description: {str(e)}")
            raise
