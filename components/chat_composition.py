class ChatComposition:
    def __init__(self, api_key, endpoint):
        self.prompt_enhancer = PromptEnhancer(api_key, endpoint)
        self.description_generator = DescriptionGenerator(api_key, endpoint)
        self.svg_generator = SVGGenerator(api_key, endpoint)
    
    async def process_request(self, user_prompt):
        try:
            # Step 1: Enhance the prompt
            enhanced_prompt = await self.prompt_enhancer.process(user_prompt)
            print("Enhanced Prompt:", enhanced_prompt)
            
            # Step 2: Generate description
            description = await self.description_generator.process(enhanced_prompt)
            print("Generated Description:", description)
            
            # Step 3: Generate SVG
            svg_code = await self.svg_generator.process(description)
            print("Generated SVG:", svg_code)
            
            return {
                "enhanced_prompt": enhanced_prompt,
                "description": description,
                "svg_code": svg_code
            }
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise 