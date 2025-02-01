import os
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyBLqWAiSlxihZ89McIXr8KLey7jAoydquo")  # Replace with your actual API key

# Create the model with adjusted generation configuration
generation_config = {
    "temperature": 0.7,  # Adjusted for more controlled output
    "top_p": 0.9,       # Adjusted for better diversity
    "top_k": 50,        # Increased to allow more options
    "max_output_tokens": 2048,  # Reduced to avoid exceeding limits
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Start a chat session with an empty history
chat_session = model.start_chat(
    history=[]
)

# Send a message to the chat session
response = chat_session.send_message("INSERT_INPUT_HERE")

# Print the response text
print(response.text)