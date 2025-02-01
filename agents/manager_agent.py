from utils.logging_config import setup_logging
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
import logging
import time
from litellm import RateLimitError

logger = setup_logging()

load_dotenv()

class ManagerAgent:
    def __init__(self, llm=None):
        self.logger = setup_logging("manager_agent")
        self.logger.info("Initializing Manager Agent...")
        
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
            role="Design Review Manager",
            goal="Examine testimonial designs focusing on aesthetics, readability, and visual harmony.",
            backstory="""You are a Design Review Manager specializing in SVG testimonial analysis. 
            Your role is to examine testimonial designs like a human would, focusing on aesthetics, readability, and visual harmony.

            CORE ANALYSIS CAPABILITIES:

            1. VISUAL INSPECTION
            Check for:
            - Overall visual appeal and professional look
            - Balance and whitespace usage
            - Typography hierarchy and readability
            - Color harmony and contrast
            - Element spacing and positioning
            - Design consistency

            2. USABILITY ASSESSMENT
            Evaluate:
            - Text readability at different sizes
            - Visual hierarchy effectiveness
            - Information flow and scanning patterns
            - Mobile/responsive considerations
            - Accessibility concerns

            3. ISSUE IDENTIFICATION
            Look for:
            - Alignment problems
            - Spacing inconsistencies 
            - Typography issues
            - Color contrast problems
            - Layout imbalances
            - Element overlaps

            4. SOLUTION RECOMMENDATIONS
            Provide:
            - Clear issue description
            - Specific fix recommendation
            - Visual improvement suggestions
            - Implementation guidance
            - Best practice alignment

            OUTPUT FORMAT:
            {
                "issues": [
                    {
                        "severity": "high|medium|low",
                        "category": "alignment|typography|spacing|color|layout",
                        "description": str,
                        "impact": str,
                        "solution": str,
                        "best_practice": str
                    }
                ],
                "summary": {
                    "total_issues": int,
                    "critical_fixes": list,
                    "visual_score": int,
                    "canvas_size": {"width": 1080, "height": 1080}
                }
            }

            EVALUATION CRITERIA:
            1. Professional Appearance
            - Clean layout
            - Balanced composition
            - Proper spacing
            - Clear hierarchy

            2. Readability
            - Font size appropriateness
            - Color contrast
            - Text spacing
            - Line height

            3. Technical Quality
            - Element alignment
            - Consistent spacing
            - Proper containment
            - Responsive design

            RESPONSE GUIDELINES:
            1. Analyze like a human viewer first
            2. Identify issues that impact visual quality
            3. Provide clear, actionable solutions
            4. Reference design best practices
            5. Focus on impactful improvements
            """,
            llm=llm,
            verbose=True
        )
        logger.info("✓ Manager Agent successfully initialized with role and goals")

    def analyze_request(self, request):
        self.logger.info(f"Analyzing request: {request}")
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
        result_str = str(result.raw_output if hasattr(result, 'raw_output') else result)
        self.logger.info(f"Analysis result: {result_str}")
        return result

    def review_svg(self, svg_code):
        """Review generated SVG and provide feedback"""
        logger.info("Starting SVG review...")
        try:
            task = Task(
                description=f"""
                You are a **Vision Agent** specializing in analyzing **SVG testimonial designs** by examining both the **image rendering** and the **SVG code**. Your job is to **identify alignment, structural, and styling issues** within SVG-based testimonials and provide actionable solutions.

                ## **Key Responsibilities & Workflow**
                ### **1. Analyze the SVG Image for Visual Issues**
                Your first task is to **visually inspect** the testimonial image and identify issues such as:
                :white_tick: **Text Alignment Issues:**
                   - The testimonial text is **misaligned inside the container** (too close to the edge, overlapping, or overflowing).
                   - The **client's name and designation** are not properly positioned.
                :white_tick: **Container & Shape Issues:**
                   - The **testimonial container shape** (rectangle, ellipse, hexagon, etc.) is **distorted, missing, or overlapping other elements**.
                   - The container **does not enclose the text properly** or has incorrect dimensions.
                :white_tick: **Unknown or Broken Shapes:**
                   - Extra, unintended shapes appear in the design.
                   - Shapes **clip into text** or interfere with the readability of testimonials.
                :white_tick: **Font & Readability Issues:**
                   - Text **goes out of the container or overflows** beyond boundaries.
                   - Font size is **too small or too large**, making it hard to read.
                   - Insufficient contrast between text and background.
                :white_tick: **Spacing & Layout Issues:**
                   - Elements are **too close or too far apart**, making the design unbalanced.
                   - The **avatar or decorative elements** are incorrectly positioned.
                   - The **testimonial title is misaligned** or overlaps with other content.
                :white_tick: **SVG Rendering Errors:**
                   - Missing elements when rendering the SVG.
                   - Shapes or texts not appearing as expected.

                ### **2. Inspect the SVG Code for Structural & Syntax Errors**
                Once visual issues are identified, analyze the **SVG code** to determine the exact causes and **propose fixes**:
                :white_tick: **Incorrect `x`, `y`, `width`, `height` Values:**
                   - Check whether **text or container elements have incorrect coordinates** causing misalignment.
                   - Ensure elements are **properly enclosed** within the designated areas.
                :white_tick: **Text Overflows & Line Break Issues:**
                   - Ensure `tspan` elements are used correctly for multi-line text.
                   - Adjust `dy` spacing for **better readability** in multi-line testimonials.
                :white_tick: **Missing or Incorrect `text-anchor` Values:**
                   - If text is centered, ensure `text-anchor="middle"` is correctly applied.
                   - For left-aligned or right-aligned layouts, verify text alignment attributes.
                :white_tick: **Broken or Unnecessary `<g>` Groups:**
                   - Detect if unnecessary grouping elements cause unintended layering issues.
                :white_tick: **Stroke & Fill Issues:**
                   - Verify that elements have proper `fill` and `stroke` attributes to **maintain visibility**.
                   - Check `opacity` values to prevent unexpected transparency problems.
                :white_tick: **Clipping & Masking Issues:**
                   - Identify if elements are unintentionally clipped due to `clipPath` or `mask` errors.

                ## **3. Generate an Issue List & Solutions**
                After identifying issues in both the **visual image** and the **SVG code**, provide a **detailed issue list** along with **precise solutions**:
                ### **Issue List Example**
                **:x: Issue 1: Text is overflowing beyond the testimonial container.**
                   - **Cause:** The testimonial text has incorrect `width` or lacks `word-wrap`.
                   - **Fix:** Set `text-wrap="balance"` and adjust `width` inside `<text>` elements.
                **:x: Issue 2: Client's name is misaligned with the testimonial text.**
                   - **Cause:** Incorrect `x`, `y` positioning of the client's name.
                   - **Fix:** Adjust `x="same as testimonial container"` and use `dy="20px"` for better spacing.
                **:x: Issue 3: Extra unknown shape appears behind the testimonial box.**
                   - **Cause:** An unintended `<rect>` or `<circle>` element exists in the SVG.
                   - **Fix:** Remove unnecessary `<rect>` elements or adjust `z-index` using `g` layers.

                ## **4. Output Final Review & Debugged SVG Code**
                Once the issues are identified, **output a debugged version of the SVG code** with all fixes applied. Ensure:
                :white_tick: **Text is properly aligned inside the container.**
                :white_tick: **Elements have correct dimensions and spacing.**
                :white_tick: **All unnecessary or broken elements are removed.**
                :white_tick: **The final SVG is visually balanced, readable, and error-free.**

                ## **Final Deliverables from the Vision Agent:**
                1. **A detailed issue list with precise explanations & fixes.**
                2. **A corrected version of the SVG code with proper alignment & formatting.**
                3. **A final validation check to confirm that the new SVG is optimized.**

                ### **Strict Guidelines for Issue Detection:**
                - **No repeating errors should go unnoticed.**
                - **Ensure that each solution provided is clear, correct, and directly applicable.**
                - **Avoid generic suggestions—give precise code fixes based on the actual problem.**

                Please analyze the provided SVG code and rendered image to identify any issues and provide actionable solutions.
                """,
                expected_output="Detailed analysis of SVG code issues and recommendations",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            logger.info("Executing SVG review task...")
            result = crew.kickoff()
            logger.info("✓ SVG review completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error reviewing SVG: {str(e)}")
            raise

    def some_method(self):
        response = self.agent.run("Your prompt here")
        return response
