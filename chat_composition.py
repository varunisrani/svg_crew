import openai
import asyncio
import json
import httpx
from typing import Optional, Dict, Any
import logging
import cairosvg  # For SVG to PNG conversion
import base64  # For encoding PNG to base64
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ChatComposition:
    def __init__(self):
        # Configure Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Using gemini-pro model
        
        # Other configurations
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.chat_model = "gpt-4o-mini"
        self.description_model = "ft:gpt-4o-mini-2024-07-18:personal::AtrjXCvd"
        self.svg_model = "ft:gpt-4o-mini-2024-07-18:personal::ArguXr7z"
        self.max_tokens = 2000
        self.temperature = 0.7
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
    async def enhance_prompt(self, user_input: str) -> Optional[str]:
        """Step 1: Enhance the user's prompt with more details"""
        logging.info(f"Step 1: Enhancing prompt for user input: '{user_input}'")
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,  # Using chat model
                messages=[
                    {"role": "system", "content": """You are an AI prompt enhancer that takes user input and refines it into a structured, visually rich, and professional description for an AI design generator.

Your objective is to enhance user-generated prompts by making them more detailed, structured, and optimized for generating high-quality SVG designs. Ensure clarity, alignment, aesthetics, and functionality while keeping the prompt concise.

Follow these instructions:

1. Expand Visual Details:
- Extract key elements (background, shape, layout, color, typography)
- Ensure well-defined visual structure
- Maintain proper alignment and hierarchy

2. Improve Readability & Clarity:
- Use precise yet descriptive language
- Avoid excessive words while retaining necessary details
- Structure information logically

3. Enhance Aesthetic Appeal:
- Specify color harmony (contrast, complementary colors)
- Suggest modern and minimalistic design principles
- Include visual effects where appropriate

4. Maintain Professional Tone:
- Write as if a designer is giving structured instructions
- Use concise and impactful wording
- Focus on design best practices

Example transformation:
User: "Create a testimonial with black background and pink circle"
Enhanced: "Generate a sleek, modern testimonial design with a deep black background for contrast and a soft pink circular container to highlight the review. Ensure balanced typography with clean spacing for readability, maintaining a professional and elegant aesthetic."
"""},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            enhanced_prompt = response.choices[0].message.content
            logging.info(f"Enhanced prompt: '{enhanced_prompt}'")
            return enhanced_prompt
        except Exception as e:
            logging.error(f"Error enhancing prompt: {e}")
            return None

    async def generate_description(self, enhanced_prompt: str) -> Optional[str]:
        """Step 2: Generate a detailed description from the enhanced prompt"""
        logging.info("Step 2: Generating description from enhanced prompt")
        try:
            response = self.client.chat.completions.create(
                model=self.description_model,  # Using description model
                messages=[
                    {"role": "system", "content": """Highly Detailed SVG Testimonial Description Generator
Your task is to generate a highly detailed, structured, and visually compelling description of an SVG testimonial design based on the given specifications. Each description must be unique, ensuring fresh and original design concepts while maintaining perfect alignment and balance.

1. Alignment Structure
Before defining the design elements, ensure that the SVG structure follows a well-balanced alignment system:
Center Alignment: All elements (title, container, text, decorations) are centrally positioned for a balanced look.
Left Alignment: The testimonial text and supporting elements are aligned to the left while maintaining visual harmony.
Right Alignment: The layout is right-aligned, ensuring elegance and readability.
Custom Alignment: A unique arrangement based on specific artistic or functional requirements.
This alignment structure ensures flexibility while maintaining aesthetic consistency.

Instructions for Description Generation
2. General Overview
Begin with an engaging introduction, describing the aesthetic theme and visual tone of the SVG design (e.g., modern, elegant, minimalistic, corporate, playful).
Mention how the background color, gradient, or texture enhances the design's visual hierarchy and usability.
Ensure that the design is aesthetically balanced and aligns with user requirements.

3. Title Placement & Styling
Define the exact x, y coordinates for the title to ensure perfect alignment.
Specify the font style, size, weight, and color, ensuring it matches the design's tone.
Use text-anchor="middle" when necessary for perfect horizontal centering.
Explain how the title's placement contributes to visual clarity and hierarchy.

4. Testimonial Container (Main Box Design)
Provide precise coordinates (x, y) for the testimonial container.
Define width, height, border-radius (rx, ry) for smooth rounded corners.
Mention the background color, gradient, shadow effects, or border enhancements for a visually distinct container.
Describe how the container's positioning, size, and styling enhance balance, readability, and responsiveness.

5. Testimonial Text Structure & Readability
Define the x, y position, font family, size, weight, and color for the testimonial text.
Use span elements for multi-line text formatting while ensuring proper spacing using dy values.
Describe the font style (bold, italic, serif, sans-serif) and its impact on readability.
Ensure that the testimonial text is perfectly aligned within the main container.

6. Decorative Elements & Enhancements
Include elements like quotation marks, icons, profile images, separators, or stars.
Define their x, y positions, sizes, colors, and how they subtly enhance the design.
Ensure these elements complement the design without cluttering the layout.

7. Client Information Placement (Name & Designation)
Specify the exact x, y coordinates for the client's name and designation.
Use an appropriate font style and size that aligns with the overall theme.
Ensure that the alignment of name and designation maintains readability and elegance.

8. Final Aesthetic Review
Summarize how color harmony, font choices, spacing, and alignment create a polished look.
Highlight how the design maintains a balance between aesthetics and functionality.
Ensure the description is unique, creative, and visually engaging, avoiding repetitive designs.

Example SVG Description (Each Generation Must Be Unique & Well-Aligned)
Modern Blue-Themed Testimonial SVG Design
This sleek, modern testimonial SVG design features a cool gradient background transitioning from deep navy blue (#0D1B2A) at the top to soft teal blue (#1B6CA8) at the bottom. The smooth gradient adds depth, creating a professional yet inviting feel.

At the top center, the title "What Our Clients Say" is placed at x="500", y="100", using a bold sans-serif font (size: 42px, weight: 700, color: #FFFFFF). The text-anchor="middle" ensures the title remains perfectly centered, reinforcing a strong visual hierarchy.

The testimonial container is positioned at x="100", y="180", with a width of 800px and a height of 350px. The container has smooth rounded corners (rx="15", ry="15") and a subtle shadow effect (rgba(0, 0, 0, 0.2)). The background color is soft white (#F9F9F9), ensuring high contrast against the dark background.

Inside the container, the testimonial text is placed at x="140", y="240", in an elegant serif font (size: 28px, weight: 400, color: #333333). To ensure proper spacing, span elements are used with dy="35", allowing for seamless multi-line formatting.

Decorative quotation marks (x="120", y="220") in a subtle gold accent (#FFD700) add a refined touch. A circular profile placeholder (radius=80px) is positioned at x="820", y="260", allowing space for user avatars.

At the bottom right of the container, the client's name and designation appear at x="650", y="500", styled in an italic serif font (size: 24px, color: #555555), adding a signature-like feel.

This design's clean typography, balanced layout, and well-placed elements make it ideal for corporate testimonials, service reviews, or portfolio highlights, ensuring a professional and aesthetically pleasing presentation."""},
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            description = response.choices[0].message.content
            logging.info(f"Generated description: '{description[:100]}...'")
            return description
        except Exception as e:
            logging.error(f"Error generating description: {e}")
            return None

    async def generate_svg(self, description: str) -> Optional[str]:
        """Step 3: Generate SVG code from the description"""
        logging.info("Step 3: Generating SVG from description")
        try:
            response = self.client.chat.completions.create(
                model=self.svg_model,
                messages=[
                    {"role": "system", "content": """You are a design assistant that helps create beautiful SVG testimonial designs. Follow these strict guidelines:

Color Palettes:
- Beige Minimalist: #E8E1D9 (bg), #9B4F3E (accent)
- Teal Modern: #65A7A1 (bg), #F5EBE4 (container)
- White & Pink: #FFFFFF (bg), #F3E5F5 (circle)
- Monochrome: #B0B0B0 (bg), #FFFFFF (container)
- Yellow & White: #E7B831 (bg), #FFFFFF (container)

Typography Rules:
- Sans-serif fonts: Arial, Open Sans, Poppins
- Serif fonts: Georgia, Playfair Display, Times New Roman
- Bold fonts: Arial Black, Impact
- Font sizes: 36-48px for body, 80-120px for titles
- Text alignment: center-aligned with text-anchor="middle"
- Line spacing: 50-60px between lines using tspan dy

Layout Structure:
- Canvas size: Always 1080x1080 pixels
- Centered content with x="540"
- Container shapes: circles, rectangles with rounded corners
- Optional decorative elements: lines, borders
- Text containers: white/light backgrounds for readability

Design Elements:
- Background fills with solid colors
- Geometric containers (circles, rectangles)
- Two-tone color schemes
- Clean typography hierarchy
- Consistent spacing
- Optional decorative accents

Always maintain:
1. 1080x1080 canvas size
2. Centered layout
3. Readable typography
4. Clean, minimal design
5. Professional color combinations
6. Proper SVG structure with comments

Example SVGs for reference:
<svg width="1080" height="1080" xmlns="http://www.w3.org/2000/svg">
  <!-- Light blue background -->
  <rect width="1080" height="1080" fill="#87CEFA"/>

  <!-- Title -->
  <text x="100" y="150"
        font-family="Arial Black, sans-serif"
        font-size="72"
        font-style="italic"
        fill="#000000">
    Peter England
  </text>

  <!-- Pink rounded rectangle for "Customer Review" -->
  <rect x="100" y="200"
        width="500" height="120"
        rx="60" ry="60"
        fill="#FFB6C1"/>
  <text x="130" y="270"
        font-family="Arial, sans-serif"
        font-size="48"
        fill="#FFFFFF">
    Customer
  </text>
  <text x="340" y="270"
        font-family="Arial, sans-serif"
        font-size="48"
        fill="#FFFFFF">
    Review
  </text>

  <!-- White review card -->
  <rect x="100" y="350"
        width="880" height="400"
        rx="20" ry="20"
        fill="#FFFFFF"/>

  <!-- Star rating -->
  <g transform="translate(400, 400)">
    <!-- Filled stars -->
    <path d="M0,0 L10,20 L0,40 L20,30 L40,40 L30,20 L40,0 L20,10 Z"
          fill="#FFD700" transform="scale(0.8)"/>
    <path d="M0,0 L10,20 L0,40 L20,30 L40,40 L30,20 L40,0 L20,10 Z"
          fill="#FFD700" transform="translate(40,0) scale(0.8)"/>
    <path d="M0,0 L10,20 L0,40 L20,30 L40,40 L30,20 L40,0 L20,10 Z"
          fill="#FFD700" transform="translate(80,0) scale(0.8)"/>
    <path d="M0,0 L10,20 L0,40 L20,30 L40,40 L30,20 L40,0 L20,10 Z"
          fill="#FFD700" transform="translate(120,0) scale(0.8)"/>
    <!-- Empty star -->
    <path d="M0,0 L10,20 L0,40 L20,30 L40,40 L30,20 L40,0 L20,10 Z"
          fill="#D3D3D3" transform="translate(160,0) scale(0.8)"/>
  </g>

  <!-- Review text -->
  <text x="140" y="480"
        font-family="Times New Roman, serif"
        font-size="24"
        fill="#000000"
        width="800">
    <tspan x="140" dy="0">"Absolutely love the quality and style of this brand! The</tspan>
    <tspan x="140" dy="30">fabric feels premium, and the designs are so unique. I've</tspan>
    <tspan x="140" dy="30">received countless compliments every time I wear their</tspan>
    <tspan x="140" dy="30">clothes. Highly recommend for anyone looking to elevate</tspan>
    <tspan x="140" dy="30">their wardrobe!"</tspan>
  </text>

  <!-- Reviewer name -->
  <text x="540" y="680"
        font-family="Arial, sans-serif"
        font-size="32"
        font-style="italic"
        text-anchor="middle"
        fill="#000000">
    Nalin Rajpal
  </text>

  <!-- Black rounded rectangle for website -->
  <rect x="100" y="800"
        width="880" height="80"
        rx="40" ry="40"
        fill="#000000"/>
  <text x="540" y="850"
        font-family="Arial, sans-serif"
        font-size="24"
        text-anchor="middle"
        fill="#FFFFFF">
    peterengland.org.com
  </text>
</svg>"""},
                    {"role": "user", "content": description}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract SVG code from the response
            content = response.choices[0].message.content
            logging.info("Raw response from model:")
            logging.info(content)
            
            # Find SVG code between xml tags
            svg_match = re.search(r'<svg.*?</svg>', content, re.DOTALL)
            if svg_match:
                svg = svg_match.group(0)
                logging.info("Successfully extracted SVG code:")
                logging.info(svg)
                return svg
            else:
                logging.error("No SVG code found in response")
                return None
            
        except Exception as e:
            logging.error(f"Error generating SVG: {e}")
            return None

    async def convert_svg_to_png(self, svg: str) -> Optional[str]:
        """Convert SVG to PNG and return base64 encoded string"""
        try:
            logging.info("Converting SVG to PNG")
            logging.info("Input SVG:")
            logging.info(svg)
            
            # Clean up SVG code
            svg = svg.strip()
            if not svg.startswith('<?xml'):
                svg = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg
            
            # Save SVG to temporary file
            with open("temp.svg", "w", encoding='utf-8') as f:
                f.write(svg)
            
            # Convert SVG to PNG using cairosvg
            png_data = cairosvg.svg2png(
                url="temp.svg",
                output_width=1080,
                output_height=1080
            )
            
            # Encode PNG to base64
            png_base64 = base64.b64encode(png_data).decode('utf-8')
            
            logging.info("Successfully converted SVG to PNG")
            return png_base64
        except Exception as e:
            logging.error(f"Error converting SVG to PNG: {e}")
            logging.error(f"SVG that caused error: {svg}")
            return None

    async def optimize_svg(self, svg: str, png_base64: str) -> Optional[str]:
        """Step 4: Optimize the SVG using Gemini model"""
        logging.info("Step 4: Optimizing SVG")
        try:
            # Create the vision prompt
            vision_prompt = f"""You are a **Vision Agent** specializing in analyzing **SVG testimonial designs** by examining both the **image rendering** and the **SVG code**. Your job is to **identify alignment, structural, and styling issues** within SVG-based testimonials and provide actionable solutions.
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

Here's the SVG code and PNG rendering to analyze:

SVG Code:
{svg}

PNG Rendering (base64):
{png_base64}

Please provide ONLY a detailed issue list with solutions. Do not include SVG code in your response. Focus on making the testimonial look visually appealing and professional."""

            # Generate response using Gemini
            try:
                response = self.gemini_model.generate_content(vision_prompt)
                
                if response and hasattr(response, 'text'):
                    logging.info("Successfully generated optimization analysis")
                    return response.text
                else:
                    logging.error("Invalid response from Gemini")
                    return None
                
            except Exception as e:
                logging.error(f"Gemini API error: {str(e)}")
                return None
            
        except Exception as e:
            logging.error(f"Error in optimize_svg: {str(e)}")
            return None

    async def enhance_visuals(self, svg: str, issue_list: str, png_base64: str) -> Optional[str]:
        """Step 5: Enhance visual elements using Gemini model"""
        logging.info("Step 5: Enhancing visual elements")
        try:
            # Create the edit prompt
            edit_prompt = f"""### **System Prompt – Edit SVG Agent**
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

Here's the current SVG to enhance:

SVG Code:
{svg}

PNG Rendering (base64):
{png_base64}

Issues Identified:
{issue_list}

Please provide:
1. A structured list of improvements made
2. Visual impact assessment
3. Accessibility compliance status
4. The enhanced SVG code that addresses all issues"""

            # Generate response using Gemini
            try:
                response = self.gemini_model.generate_content(edit_prompt)
                
                if response and hasattr(response, 'text'):
                    logging.info("Successfully enhanced SVG")
                    return response.text
                else:
                    logging.error("Invalid response from Gemini")
                    return None
                
            except Exception as e:
                logging.error(f"Gemini API error: {str(e)}")
                return None
            
        except Exception as e:
            logging.error(f"Error in enhance_visuals: {str(e)}")
            return None

    async def run_svg_crew(self, svg_code: str, png_base64: str) -> Dict[str, Any]:
        """Run the SVG Crew process with Vision, Edit and CEO agents"""
        logging.info("Starting SVG Crew analysis")
        
        try:
            # Vision Agent Analysis
            vision_prompt = f"""You are a Vision Agent specializing in analyzing SVG testimonial designs. 
            Here's the SVG code and PNG rendering to analyze:

            SVG Code:
            {svg_code}

            PNG Rendering:
            {png_base64}

            Please analyze for:
            1. Visual layout and alignment issues
            2. Text readability and spacing
            3. Color and contrast effectiveness
            4. Overall aesthetic balance
            
            Provide a detailed analysis with specific issues and solutions."""

            vision_response = await asyncio.to_thread(
                self.vision_agent.generate_content,
                vision_prompt
            )
            
            # Edit Agent Implementation
            edit_prompt = f"""You are an Edit SVG Agent. Based on the Vision Agent's analysis:
            {vision_response.text}

            Here's the SVG to optimize:
            {svg_code}

            Please provide:
            1. Corrected SVG code
            2. List of improvements made
            3. Visual impact assessment"""

            edit_response = await asyncio.to_thread(
                self.edit_agent.generate_content,
                edit_prompt
            )
            
            # CEO Agent Validation
            ceo_prompt = f"""You are the CEO Agent responsible for final validation.
            
            Original SVG:
            {svg_code}
            
            Vision Analysis:
            {vision_response.text}
            
            Edit Implementation:
            {edit_response.text}
            
            Please provide:
            1. Final assessment of changes
            2. Approval or revision requests
            3. Quality assurance report"""

            ceo_response = await asyncio.to_thread(
                self.ceo_agent.generate_content,
                ceo_prompt
            )
            
            return {
                "vision_analysis": vision_response.text,
                "edit_implementation": edit_response.text,
                "ceo_validation": ceo_response.text
            }
            
        except Exception as e:
            logging.error(f"Error in SVG Crew process: {e}")
            return {
                "error": str(e)
            }

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process the entire request through all steps"""
        logging.info("Starting testimonial generation process")
        result = {
            "enhanced_prompt": None,
            "description": None,
            "svg": None,
            "png_base64": None,
            "optimized_svg": None,
            "final_svg": None,
            "error": None
        }
        
        try:
            # Steps 1-3 (existing)
            enhanced_prompt = await self.enhance_prompt(user_input)
            if not enhanced_prompt:
                raise Exception("Failed to enhance prompt")
            result["enhanced_prompt"] = enhanced_prompt
            
            description = await self.generate_description(enhanced_prompt)
            if not description:
                raise Exception("Failed to generate description")
            result["description"] = description
            
            svg = await self.generate_svg(description)
            if not svg:
                raise Exception("Failed to generate SVG")
            result["svg"] = svg
            
            # Convert SVG to PNG
            png_base64 = await self.convert_svg_to_png(svg)
            if not png_base64:
                raise Exception("Failed to convert SVG to PNG")
            result["png_base64"] = png_base64
            
            # Step 4: Optimize SVG with both SVG and PNG
            optimized_result = await self.optimize_svg(svg, png_base64)
            if not optimized_result:
                raise Exception("Failed to optimize SVG")
            
            # Extract issue list from step 4 result
            try:
                # Parse the response to get just the issue list
                parts = optimized_result.split("=== Final Review ===")
                issue_list = parts[1].strip() if len(parts) > 1 else ""
                
                # Store optimized SVG for reference but don't use it in step 5
                result["optimized_svg"] = optimized_result
                
                # Step 5: Enhance visuals using original SVG and issue list
                final_svg = await self.enhance_visuals(svg, issue_list, png_base64)  # Using original svg instead of optimized
                if not final_svg:
                    raise Exception("Failed to enhance visuals")
                result["final_svg"] = final_svg
                
            except Exception as e:
                logging.error(f"Error processing optimization result: {e}")
                raise Exception("Failed to process optimization result")
            
            # Run SVG Crew process after initial SVG generation
            crew_results = await self.run_svg_crew(svg, png_base64)
            if "error" not in crew_results:
                result.update(crew_results)
            
            logging.info("Testimonial generation and SVG Crew analysis completed successfully")
            
        except Exception as e:
            error_msg = f"Error processing request: {e}"
            logging.error(error_msg)
            result["error"] = str(e)
            
        return result

async def main():
    # Get user input
    print("\n=== SVG Testimonial Generator ===")
    print("Please enter your testimonial prompt (e.g., 'Create a testimonial card with blue background'):")
    user_input = input("> ")
    
    try:
        # Initialize chat composition
        chat = ChatComposition()
        
        # Process the request
        print("\nProcessing your request...")
        result = await chat.process_request(user_input)
        
        # Display results
        print("\n=== Results ===")
        if result["error"]:
            print(f"\nError: {result['error']}")
        else:
            print("\n1. Enhanced Prompt:")
            print("-" * 80)
            print(result["enhanced_prompt"])
            
            print("\n2. Generated Description:")
            print("-" * 80)
            print(result["description"])
            
            print("\n3. Generated SVG:")
            print("-" * 80)
            print(result["svg"])
            
            print("\n4. Optimization Analysis:")
            print("-" * 80)
            print(result["optimized_svg"])
            
            print("\n5. Final Enhanced SVG:")
            print("-" * 80)
            print(result["final_svg"])
            
            # Save the final SVG to a file
            if result["final_svg"]:
                filename = "output.svg"
                with open(filename, "w") as f:
                    f.write(result["final_svg"])
                print(f"\nFinal SVG saved to {filename}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Run the main function
    asyncio.run(main())

