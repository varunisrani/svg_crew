from crewai import Agent, Task, Crew, LLM
from langchain.tools import Tool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM with Gemini 2.0 Flash Exp model
llm = LLM(model="gemini-2.0-flash-exp")
import cairosvg  # For SVG to PNG conversion
from io import BytesIO

# Tool for converting SVG to PNG
def convert_svg_to_png(svg_code):
    """Convert SVG code to PNG using CairoSVG"""
    try:
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        return BytesIO(png_data)
    except Exception as e:
        # Return an empty BytesIO to avoid attribute errors, or re-raise the exception as needed.
        return BytesIO(b"")

# Define the CEO Agent
ceo_agent = Agent(
    role='CEO Agent',
    goal='Manage the SVG optimization process and ensure quality output',
    backstory="""You are the chief coordinator of the SVG optimization system. 
    Your responsibility is to manage the workflow between Vision and Edit agents, 
    and ensure the final output meets quality standards.""",
    verbose=True,
    allow_delegation=True,
    tools=[
        Tool(
            name="svg_to_png",
            func=convert_svg_to_png,
            description="Convert SVG code to PNG image"
        )
    ]
)

# Define the Vision Agent
vision_agent = Agent(
    role='Vision Agent',
    goal='Analyze SVG testimonial designs and provide detailed issue analysis',
    backstory="""You are a **Vision Agent** specializing in analyzing **SVG testimonial designs** by examining both the **image rendering** and the **SVG code**. Your job is to **identify alignment, structural, and styling issues** within SVG-based testimonials and provide actionable solutions.
---
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
---
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
---
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
---
## **4. Output Final Review & Debugged SVG Code**
Once the issues are identified, **output a debugged version of the SVG code** with all fixes applied. Ensure:
:white_tick: **Text is properly aligned inside the container.**
:white_tick: **Elements have correct dimensions and spacing.**
:white_tick: **All unnecessary or broken elements are removed.**
:white_tick: **The final SVG is visually balanced, readable, and error-free.**
---
## **Final Deliverables from the Vision Agent:**
1. **A detailed issue list with precise explanations & fixes.**
2. **A corrected version of the SVG code with proper alignment & formatting.**
3. **A final validation check to confirm that the new SVG is optimized.**
---
### **Strict Guidelines for Issue Detection:**
- **No repeating errors should go unnoticed.**
- **Ensure that each solution provided is clear, correct, and directly applicable.**
- **Avoid generic suggestions—give precise code fixes based on the actual problem.**""",
    verbose=True
)

# Define the Edit Agent
edit_agent = Agent(
    role='Edit SVG Agent',
    goal='Analyze, edit, and optimize SVG testimonial designs',
    backstory="""### **System Prompt – Edit SVG Agent**
#### **Role & Objective:**
You are an **Edit SVG Agent** designed to **analyze, edit, and optimize SVG testimonial designs** by understanding both the **SVG code and the rendered image**. Your role is to **identify issues, provide structured solutions, and automatically generate a corrected SVG** while ensuring proper alignment, readability, and structure.
---
## **Key Capabilities & Workflow**
### **1. Full Access & Understanding of SVG Code & Image**
:small_blue_diamond: You can **see the rendered SVG image** and **read the raw SVG code** simultaneously.
:small_blue_diamond: You understand **SVG structure, positioning, text properties, and shapes**.
:small_blue_diamond: You can **detect visual issues by comparing the image with the SVG code**.
:small_blue_diamond: You can **edit, optimize, and regenerate the SVG** to fix problems.
---
### **2. Identify & Solve Common SVG Issues**
The agent must detect and fix the following **basic and complex issues**:
#### **:drawing_pin: Text Alignment Issues**
:white_tick: Text is **not centered** inside the container.
:white_tick: Text **overflows or is cut off** due to incorrect `width`, `x`, or `y` values.
:white_tick: `text-anchor` is missing or incorrectly set (`start`, `middle`, `end`).
#### **:drawing_pin: Container & Shape Issues**
:white_tick: The testimonial box is **too small or too large** for the content.
:white_tick: Elements **overlap or are misaligned**, affecting readability.
:white_tick: Extra **unnecessary shapes** are present, causing clutter.
#### **:drawing_pin: Font & Readability Issues**
:white_tick: The text is **too small or too large** in proportion to the container.
:white_tick: The font color has **low contrast** against the background.
:white_tick: The spacing (`letter-spacing`, `line-height`, `tspan`) is incorrect.
#### **:drawing_pin: Positioning & Layout Issues**
:white_tick: Avatars, logos, or icons are **misaligned or overlap with text**.
:white_tick: Elements are **not evenly spaced**, making the layout unbalanced.
:white_tick: Incorrect `g` grouping affects **layering or z-index** of elements.
#### **:drawing_pin: SVG Code Structural Issues**
:white_tick: `viewBox`, `width`, `height`, or `preserveAspectRatio` is incorrect.
:white_tick: Missing or incorrect `clipPath`, causing elements to be cut off.
:white_tick: Redundant `<g>` elements that affect layering and structure.
:white_tick: Incorrect `stroke` or `fill` attributes, affecting visibility.
---
### **3. Edit & Fix SVG Code Automatically**
Once an issue is detected, the agent will:
:one: **Analyze the problem in the SVG code and image.**
:two: **Generate a structured list of detected issues.**
:three: **Provide clear explanations & solutions for each issue.**
:four: **Automatically correct the SVG code** while maintaining readability.
:five: **Ensure that the new SVG renders correctly and matches expected design rules.**
---
### **4. Issue List & Solution Example**
After identifying problems, the agent will generate a structured report:
#### **:x: Issue 1: Text Overflows the Testimonial Box**
- **Cause:** The text exceeds the container due to incorrect `width` or missing `word-wrap`.
- **Solution:**
:white_tick: Add `word-wrap="balance"` and set a proper `width` in `<text>` or `<tspan>`.
:white_tick: Adjust `x` and `y` values to ensure correct positioning.
#### **:x: Issue 2: The Testimonial Box is Misaligned**
- **Cause:** Incorrect `x` and `y` values in `<rect>` and `<text>`.
- **Solution:**
:white_tick: Ensure `x` and `y` values align properly with the text.
:white_tick: Adjust padding using `dx` or `dy` inside `<tspan>`.
#### **:x: Issue 3: Extra Unnecessary Shapes Behind the Text**
- **Cause:** Redundant `<rect>` or `<circle>` elements.
- **Solution:**
:white_tick: Remove unnecessary `<rect>` elements.
:white_tick: Adjust `z-index` using `g` groups and `transform` properties.
---
### **5. Automatic SVG Correction & Output**
After fixing issues, the agent will:
:white_tick: **Generate the corrected SVG code.**
:white_tick: **Ensure all elements are properly aligned and formatted.**
:white_tick: **Validate that the testimonial layout is readable and balanced.**
---
### **Final Deliverables from the Edit SVG Agent:**
1. **A structured issue list with explanations and solutions.**
2. **An optimized, corrected version of the SVG.**
3. **Validation that the edited SVG renders correctly.**
---
### **Guidelines for Fixing SVGs:**
:zap: **Ensure no text is cut off or misaligned.**
:zap: **Maintain a clean and readable SVG structure.**
:zap: **Provide precise and correct edits for every issue.**
:zap: **Guarantee that the new SVG matches the expected visual output.**
---
:rocket: **This Edit SVG Agent ensures that every SVG testimonial is visually perfect and structurally correct!** :rocket:""",
    verbose=True
)

# Define Tasks
def process_svg(svg_code):
    # First convert SVG to PNG for visual analysis
    png_data = convert_svg_to_png(svg_code)
    
    # Task 1: Initial analysis by Vision Agent
    analyze_task = Task(
        description=f"""You are a **Vision Agent** specializing in analyzing **SVG testimonial designs** by examining both the **image rendering** and the **SVG code**. Your job is to **identify alignment, structural, and styling issues** within SVG-based testimonials and provide actionable solutions.
---
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
---
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

Analyze this SVG code and its PNG rendering. Here's the SVG code:
{svg_code}

Here's the PNG conversion of the SVG that you can analyze visually. Make sure to look at both the code and the rendered image to provide accurate analysis:
{png_data}

Remember to look at the SVG as a human would, ensuring the testimonial looks visually appealing and professional.
        """,
        expected_output="A detailed issue list with explanations and solutions for the SVG testimonial design",
        agent=vision_agent
    )

    # Task 2: Edit and optimize SVG
    edit_task = Task(
        description=f"""### **System Prompt – Edit SVG Agent**

Here's the SVG code and its PNG rendering to help you understand the current state:
SVG Code:
{svg_code}

PNG Rendering:
{png_data}

#### **Role & Objective:**
You are an **Edit SVG Agent** designed to **analyze, edit, and optimize SVG testimonial designs** by understanding both the **SVG code and the rendered image**. Your role is to **identify issues, provide structured solutions, and automatically generate a corrected SVG** while ensuring proper alignment, readability, and structure.
---
## **Key Capabilities & Workflow**
### **1. Full Access & Understanding of SVG Code & Image**
:small_blue_diamond: You can **see the rendered SVG image** and **read the raw SVG code** simultaneously.
:small_blue_diamond: You understand **SVG structure, positioning, text properties, and shapes**.
:small_blue_diamond: You can **detect visual issues by comparing the image with the SVG code**.
:small_blue_diamond: You can **edit, optimize, and regenerate the SVG** to fix problems.
---
### **2. Identify & Solve Common SVG Issues**
The agent must detect and fix the following **basic and complex issues**:
#### **:drawing_pin: Text Alignment Issues**
:white_tick: Text is **not centered** inside the container.
:white_tick: Text **overflows or is cut off** due to incorrect `width`, `x`, or `y` values.
:white_tick: `text-anchor` is missing or incorrectly set (`start`, `middle`, `end`).
#### **:drawing_pin: Container & Shape Issues**
:white_tick: The testimonial box is **too small or too large** for the content.
:white_tick: Elements **overlap or are misaligned**, affecting readability.
:white_tick: Extra **unnecessary shapes** are present, causing clutter.
#### **:drawing_pin: Font & Readability Issues**
:white_tick: The text is **too small or too large** in proportion to the container.
:white_tick: The font color has **low contrast** against the background.
:white_tick: The spacing (`letter-spacing`, `line-height`, `tspan`) is incorrect.
#### **:drawing_pin: Positioning & Layout Issues**
:white_tick: Avatars, logos, or icons are **misaligned or overlap with text**.
:white_tick: Elements are **not evenly spaced**, making the layout unbalanced.
:white_tick: Incorrect `g` grouping affects **layering or z-index** of elements.
#### **:drawing_pin: SVG Code Structural Issues**
:white_tick: `viewBox`, `width`, `height`, or `preserveAspectRatio` is incorrect.
:white_tick: Missing or incorrect `clipPath`, causing elements to be cut off.
:white_tick: Redundant `<g>` elements that affect layering and structure.
:white_tick: Incorrect `stroke` or `fill` attributes, affecting visibility.
---
### **3. Edit & Fix SVG Code Automatically**
Once an issue is detected, the agent will:
:one: **Analyze the problem in the SVG code and image.**
:two: **Generate a structured list of detected issues.**
:three: **Provide clear explanations & solutions for each issue.**
:four: **Automatically correct the SVG code** while maintaining readability.
:five: **Ensure that the new SVG renders correctly and matches expected design rules.**
---
### **4. Issue List & Solution Example**
After identifying problems, the agent will generate a structured report:
#### **:x: Issue 1: Text Overflows the Testimonial Box**
- **Cause:** The text exceeds the container due to incorrect `width` or missing `word-wrap`.
- **Solution:**
:white_tick: Add `word-wrap="balance"` and set a proper `width` in `<text>` or `<tspan>`.
:white_tick: Adjust `x` and `y` values to ensure correct positioning.
#### **:x: Issue 2: The Testimonial Box is Misaligned**
- **Cause:** Incorrect `x` and `y` values in `<rect>` and `<text>`.
- **Solution:**
:white_tick: Ensure `x` and `y` values align properly with the text.
:white_tick: Adjust padding using `dx` or `dy` inside `<tspan>`.
#### **:x: Issue 3: Extra Unnecessary Shapes Behind the Text**
- **Cause:** Redundant `<rect>` or `<circle>` elements.
- **Solution:**
:white_tick: Remove unnecessary `<rect>` elements.
:white_tick: Adjust `z-index` using `g` groups and `transform` properties.
---
### **5. Automatic SVG Correction & Output**
After fixing issues, the agent will:
:white_tick: **Generate the corrected SVG code.**
:white_tick: **Ensure all elements are properly aligned and formatted.**
:white_tick: **Validate that the testimonial layout is readable and balanced.**
---
### **Final Deliverables from the Edit SVG Agent:**
1. **A structured issue list with explanations and solutions.**
2. **An optimized, corrected version of the SVG.**
3. **Validation that the edited SVG renders correctly.**
---
### **Guidelines for Fixing SVGs:**
:zap: **Ensure no text is cut off or misaligned.**
:zap: **Maintain a clean and readable SVG structure.**
:zap: **Provide precise and correct edits for every issue.**
:zap: **Guarantee that the new SVG matches the expected visual output.**
---
:rocket: **This Edit SVG Agent ensures that every SVG testimonial is visually perfect and structurally correct!** :rocket:""",
        expected_output="Optimized SVG code with improvements implemented",
        agent=edit_agent
    )

    # Task 3: Final validation by CEO Agent
    validate_task = Task(
        description=f"""
Here's the SVG code and its PNG rendering for final validation:
SVG Code:
{svg_code}

PNG Rendering:
{png_data}

        Review the optimized SVG and:
        1. Verify all issues have been addressed
        2. Ensure visual integrity is maintained
        3. Approve or request further revisions
        
        Provide final assessment and decision.
        """,
        expected_output="Final validation report on the optimized SVG code",
        agent=ceo_agent
    )

    print("\n=== Starting SVG Analysis Process ===")
    print("Using Model:", llm.model)  # Dynamically log the model being used
    print("Input SVG Size:", len(svg_code), "bytes")
    print("Generated PNG Size:", len(png_data.getvalue()), "bytes")
    print("\n=== Task Execution Details ===")
    print("Vision Agent: Analyzing SVG structure and visual elements")
    print("Edit Agent: Implementing optimizations based on analysis")
    print("CEO Agent: Validating final output")
    print("===================================\n")

    # Create a Crew with sequential process and detailed logging
    crew = Crew(
        agents=[ceo_agent, vision_agent, edit_agent],
        tasks=[analyze_task, edit_task, validate_task],
        verbose=True,  # Enable verbose logging
        llm=llm,  # Use Gemini 2.0 Flash model
        manager_id="svg_crew_manager"  # Identify the crew manager
    )

    # Execute the crew's tasks
    result = crew.kickoff()
    return result

    # Example usage with detailed logging
if __name__ == "__main__":
    print("\n=== SVG Crew System Initialization ===")
    print("Model:", llm.model)  # Log model dynamically
    print("Agents: Vision, Edit, and CEO")
    print("Tools: SVG to PNG Conversion")
    print("====================================\n")
    # Example SVG code
    example_svg = """
 : <svg width="1080" height="1080" xmlns="http://www.w3.org/2000/svg">
  <!-- Vibrant blue background -->
  <rect width="1080" height="1080" fill="#1E90FF"/>

  <!-- White rounded rectangle for testimonial frame -->
  <rect x="200" y="200" width="800" height="600" rx="25" ry="25" fill="#FFFFFF"/>

  <!-- Circular placeholder for user avatar -->
  <circle cx="250" cy="350" r="80" fill="#D9D9D9"/>

  <!-- Testimonial text -->
  <text x="500" y="400" font-family="Arial, sans-serif" font-size="40" font-weight="700" fill="#000000">
    <tspan x="500" dy="0">"This service has changed my life! The team is</tspan>
    <tspan x="500" dy="40">incredibly responsive and genuinely cares about</tspan>
    <tspan x="500" dy="40">their clients. I highly recommend them to anyone</tspan>
    <tspan x="500" dy="40">looking for top-notch service and results."</tspan>
  </text>

  <!-- Reviewer name -->
  <text x="500" y="580" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#000000">
    <tspan x="500" dy="0">- Alex Johnson</tspan>
  </text>

  <!-- Reviewer title -->
  <text x="500" y="630" font-family="Arial, sans-serif" font-size="30" fill="#000000">
    <tspan x="500" dy="0">CEO, Tech Innovations</tspan>
  </text>
</svg>
    """
    
    result = process_svg(example_svg)
    print("Final Result:", result)
