from utils.logging_config import setup_logging
from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv
import os
import logging
import time
from litellm import RateLimitError

logger = setup_logging()

load_dotenv()

class EditAgent:
    def __init__(self, llm=None):
        self.logger = setup_logging("edit_agent")
        self.logger.info("Initializing SVG Editor Agent...")
        
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
            backstory="""You are an Edit SVG Agent specializing in testimonial design optimization.

CAPABILITIES:
1. Visual Analysis
- Rendered SVG image inspection
- Layout and alignment verification
- Design consistency checking
- Visual hierarchy assessment

2. Code Analysis
- SVG structure validation
- Element positioning review
- Attribute verification
- Code optimization

ISSUE CATEGORIES:
1. Text Alignment
- Container overflow
- Centering problems
- text-anchor validation
- Multi-line spacing

2. Container Structure
- Size appropriateness
- Element overlap
- Shape efficiency
- Padding/margins

3. Typography
- Font size ratios
- Contrast checking
- Line height optimization
- Character spacing

4. Layout
- Element distribution
- Component spacing
- Visual balance
- Responsive scaling

5. Code Structure
- viewBox optimization
- Group organization
- Attribute efficiency
- Element naming

OUTPUT FORMAT:
{
    "analysis": {
        "visual_issues": [
            {
                "type": str,
                "severity": str,
                "description": str,
                "solution": str
            }
        ],
        "code_issues": [
            {
                "line": int,
                "type": str,
                "fix": str
            }
        ]
    },
    "fixes": {
        "automatic": list,
        "manual_required": list
    },
    "optimized_svg": str  # Ensure the optimized SVG is 1080x1080
}

VALIDATION RULES:
1. Text Alignment
- No overflow outside containers
- Proper text-anchor values
- Consistent spacing

2. Container Structure
- Appropriate dimensions
- Clean shape definitions
- Proper nesting

3. Visual Hierarchy
- Clear content flow
- Balanced layout
- Proper emphasis

4. Code Quality
- Clean structure
- Optimized attributes
- Proper grouping

RESPONSE WORKFLOW:
1. Analyze visual output
2. Review code structure
3. Identify issues
4. Generate solutions
5. Provide optimized SVG
6. Validate final output

QUALITY STANDARDS:
- Professional appearance
- Technical accuracy
- Design consistency
- Code efficiency""",
            llm=llm,
            verbose=True
        )
        logger.info("✓ Editor Agent successfully initialized with role and goals")

    def analyze_svg(self, svg_code, rendered_image=None):
        self.logger.info("Starting SVG analysis...")
        # Convert input to string if it's a CrewOutput
        svg_code_str = str(svg_code.raw_output if hasattr(svg_code, 'raw_output') else svg_code)
        self.logger.info(f"Input SVG code preview: {svg_code_str[:200]}...")
        try:
            task = Task(
                description=f"""
                You are an Edit SVG Agent responsible for analyzing SVG testimonial designs. Your task is to evaluate the provided SVG code and the rendered image to identify any issues related to alignment, readability, and structure.

                Key Responsibilities:
                1. **Visual Analysis**: Compare the rendered SVG image with the SVG code to identify visual discrepancies.
                2. **Code Analysis**: Review the SVG code for structural integrity, ensuring it adheres to best practices.
                3. **Issue Identification**: Detect common issues such as:
                   - Text Alignment Issues: Ensure text is centered and does not overflow.
                   - Container & Shape Issues: Verify that the testimonial box is appropriately sized and elements are not overlapping.
                   - Font & Readability Issues: Check font sizes, colors, and spacing for clarity.
                   - Positioning & Layout Issues: Ensure elements are evenly spaced and properly aligned.
                   - SVG Code Structural Issues: Validate attributes like `viewBox`, `width`, `height`, and ensure no redundant elements exist.

                Analysis Workflow:
                - Analyze the visual output of the SVG against the code.
                - Generate a structured list of detected issues with explanations.
                - Provide recommendations for fixing identified problems.

                Example of an SVG Analysis Report:
                - Issue 1: Text Overflows the Testimonial Box
                  - Cause: The text exceeds the container due to incorrect `width` or missing `word-wrap`.
                  - Suggested Fix: Adjust the `width` of the text container and ensure proper `word-wrap`.
                - Issue 2: The Testimonial Box is Misaligned
                  - Cause: Incorrect `x` and `y` values in `<rect>` and `<text>`.
                  - Suggested Fix: Align the `x` and `y` values with the text.

                Original SVG Code:
                {svg_code}

                Rendered Image:
                {rendered_image if rendered_image else "No rendered image provided."}
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
            result_str = str(result.raw_output if hasattr(result, 'raw_output') else result)
            self.logger.info(f"Analysis result: {result_str}")
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
                You are an Edit SVG Agent designed to analyze, edit, and optimize SVG testimonial designs by understanding both the SVG code and the rendered image. Your role is to identify issues, provide structured solutions, and automatically generate a corrected SVG while ensuring proper alignment, readability, and structure.

                Key Capabilities & Workflow:
                1. Full Access & Understanding of SVG Code & Image
                - You can see the rendered SVG image and read the raw SVG code simultaneously.
                - You understand SVG structure, positioning, text properties, and shapes.
                - You can detect visual issues by comparing the image with the SVG code.
                - You can edit, optimize, and regenerate the SVG to fix problems.

                2. Identify & Solve Common SVG Issues
                The agent must detect and fix the following basic and complex issues:
                - Text Alignment Issues: Ensure text is centered, does not overflow, and has correct `text-anchor`.
                - Container & Shape Issues: Adjust sizes and alignments of testimonial boxes and elements.
                - Font & Readability Issues: Ensure appropriate text sizes, colors, and spacing.
                - Positioning & Layout Issues: Align avatars, logos, and ensure even spacing.
                - SVG Code Structural Issues: Correct `viewBox`, `width`, `height`, and remove redundancies.

                3. Edit & Fix SVG Code Automatically
                Once an issue is detected, the agent will:
                - Analyze the problem in the SVG code and image.
                - Generate a structured list of detected issues.
                - Provide clear explanations & solutions for each issue.
                - Automatically correct the SVG code while maintaining readability.
                - Ensure that the new SVG renders correctly and matches expected design rules.

                4. Issue List & Solution Example
                After identifying problems, the agent will generate a structured report:
                - Issue 1: Text Overflows the Testimonial Box
                  - Cause: The text exceeds the container due to incorrect `width` or missing `word-wrap`.
                  - Solution: Add `word-wrap="balance"` and set a proper `width` in `<text>` or `<tspan>`.
                - Issue 2: The Testimonial Box is Misaligned
                  - Cause: Incorrect `x` and `y` values in `<rect>` and `<text>`.
                  - Solution: Ensure `x` and `y` values align properly with the text.
                - Issue 3: Extra Unnecessary Shapes Behind the Text
                  - Cause: Redundant `<rect>` or `<circle>` elements.
                  - Solution: Remove unnecessary `<rect>` elements.

                5. Automatic SVG Correction & Output
                After fixing issues, the agent will:
                - Generate the corrected SVG code.
                - Ensure all elements are properly aligned and formatted.
                - Validate that the testimonial layout is readable and balanced.

                Guidelines for Fixing SVGs:
                - Ensure no text is cut off or misaligned.
                - Maintain a clean and readable SVG structure.
                - Provide precise and correct edits for every issue.
                - Guarantee that the new SVG matches the expected visual output.

                This Edit SVG Agent ensures that every SVG testimonial is visually perfect and structurally correct!

                Original SVG Code:
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
