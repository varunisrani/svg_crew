from crewai import Crew, Task
from dotenv import load_dotenv
import logging
import os
import time
from agents.description_agent import DescriptionAgent
from agents.svg_agent import SVGAgent
from agents.edit_agent import EditAgent
from agents.manager_agent import ManagerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SVGTestimonialCreator:
    def __init__(self):
        logger.info("Initializing SVG Testimonial Creator...")
        try:
            # Initialize all agents
            self.description_agent = DescriptionAgent()
            self.svg_agent = SVGAgent()
            self.edit_agent = EditAgent()
            self.manager_agent = ManagerAgent()
            logger.info("✓ All agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

    def create_testimonial(self, user_request):
        """
        Orchestrate the SVG testimonial creation process using all agents
        """
        try:
            # 1. Manager analyzes request
            logger.info("Starting request analysis...")
            analysis = self.manager_agent.analyze_request(user_request)
            logger.info("✓ Request analysis completed")

            # 2. Description agent generates detailed description
            logger.info("Starting SVG description generation...")
            svg_description = self.description_agent.generate_description(analysis)
            logger.info("✓ SVG description generated")

            # 3. SVG agent creates initial SVG code
            logger.info("Starting initial SVG code generation...")
            initial_svg = self.svg_agent.generate_svg(svg_description)
            logger.info("✓ Initial SVG code generated")

            # 4. Edit agent analyzes and fixes issues
            logger.info("Starting SVG analysis...")
            issues = self.edit_agent.analyze_svg(initial_svg)
            
            if issues:
                logger.info(f"Found issues in SVG, starting fixes...")
                final_svg = self.edit_agent.fix_svg(initial_svg, issues)
                logger.info("✓ SVG issues fixed")
            else:
                logger.info("No issues found in SVG")
                final_svg = initial_svg

            # 5. Manager reviews final output
            logger.info("Starting final review...")
            review = self.manager_agent.review_svg(final_svg)
            logger.info("✓ Final review completed")

            # 6. SVG agent optimizes the final code
            logger.info("Starting SVG optimization...")
            optimized_svg = self.svg_agent.optimize_svg(final_svg)
            logger.info("✓ SVG optimization completed")

            # Extract string content from CrewOutput objects
            svg_description_str = str(svg_description.raw_output if hasattr(svg_description, 'raw_output') else svg_description)
            optimized_svg_str = str(optimized_svg.raw_output if hasattr(optimized_svg, 'raw_output') else optimized_svg)
            review_str = str(review.raw_output if hasattr(review, 'raw_output') else review)

            # Save the SVG to file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/testimonial_{timestamp}.svg"
            os.makedirs("outputs", exist_ok=True)
            
            with open(filename, "w") as f:
                f.write(optimized_svg_str)
            logger.info(f"✓ SVG saved to {filename}")

            return {
                'description': svg_description_str,
                'svg_code': optimized_svg_str,
                'review': review_str,
                'filename': filename
            }

        except Exception as e:
            logger.error(f"Error in SVG creation process: {str(e)}")
            raise

def main():
    try:
        creator = SVGTestimonialCreator()
        
        # Get user input
        user_request = input("Enter your testimonial requirements: ")
        logger.info("Starting testimonial creation process...")
        
        result = creator.create_testimonial(user_request)
        
        # Print results
        print("\n=== Generated Description ===")
        print(result['description'])
        
        print("\n=== Generated SVG Code ===")
        print(result['svg_code'])
        
        print("\n=== Final Review ===")
        print(result['review'])
        
        print(f"\nSVG file saved to: {result['filename']}")
        logger.info("✓ Testimonial creation process completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to create testimonial: {str(e)}")
        raise

if __name__ == "__main__":
    main()
