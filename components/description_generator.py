from .chat_base import ChatComponent

class DescriptionGenerator(ChatComponent):
    async def process(self, enhanced_prompt):
        system_msg = """You are a Description Generator. Your task is to:
        1. Convert the enhanced prompt into a detailed visual description
        2. Specify layout, colors, typography, and visual elements
        3. Include specific measurements and positioning
        4. Consider accessibility and design principles"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        response = await self._call_api(messages)
        return response.choices[0].message.content

    async def _call_api(self, messages):
        # API call implementation
        pass 