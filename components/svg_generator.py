from .chat_base import ChatComponent

class SVGGenerator(ChatComponent):
    async def process(self, description):
        system_msg = """You are an SVG Code Generator. Your task is to:
        1. Convert the visual description into valid SVG code
        2. Follow SVG best practices
        3. Ensure accessibility
        4. Optimize the code
        5. Include appropriate comments"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": description}
        ]
        
        response = await self._call_api(messages)
        return response.choices[0].message.content

    async def _call_api(self, messages):
        # API call implementation
        pass 