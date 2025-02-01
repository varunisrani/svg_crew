from abc import ABC, abstractmethod

class ChatComponent(ABC):
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint
    
    @abstractmethod
    async def process(self, input_text):
        pass 