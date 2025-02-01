import asyncio
from chat_composition import ChatComposition
import json

async def main():
    # Initialize the chat composition
    chat = ChatComposition()
    
    # Example user input
    user_input = "Create a testimonial card with a quote"
    
    print("Processing request...")
    print(f"User input: {user_input}\n")
    
    # Process the request
    result = await chat.process_request(user_input)
    
    # Print results
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print("Enhanced Prompt:")
        print(result["enhanced_prompt"])
        print("\nGenerated Description:")
        print(result["description"])
        print("\nGenerated SVG:")
        print(result["svg"])
        
        # Save the SVG to a file
        if result["svg"]:
            filename = "output.svg"
            with open(filename, "w") as f:
                f.write(result["svg"])
            print(f"\nSVG saved to {filename}")
            
        # Save the full result to a JSON file
        with open("result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Full result saved to result.json")

if __name__ == "__main__":
    asyncio.run(main()) 