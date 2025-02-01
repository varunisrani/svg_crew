import { useState } from 'react';
import { OpenAIApi, Configuration } from 'openai';

// Initialize OpenAI client
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
});
const openai = new OpenAIApi(configuration);

export function ChatComposition() {
  const [userInput, setUserInput] = useState('');
  const [enhancedPrompt, setEnhancedPrompt] = useState('');
  const [description, setDescription] = useState('');
  const [svgOutput, setSvgOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Step 1: Enhance the user prompt
  const enhancePrompt = async (input: string) => {
    try {
      const response = await openai.createChatCompletion({
        model: "gpt-3.5-turbo",
        messages: [{
          role: "system",
          content: "You are a helpful assistant that enhances user prompts to be more detailed and specific."
        }, {
          role: "user", 
          content: input
        }]
      });
      return response.data.choices[0].message?.content;
    } catch (error) {
      console.error('Error enhancing prompt:', error);
      throw error;
    }
  };

  // Step 2: Generate description from enhanced prompt
  const generateDescription = async (enhancedPrompt: string) => {
    try {
      const response = await openai.createChatCompletion({
        model: "gpt-3.5-turbo",
        messages: [{
          role: "system",
          content: "You are a descriptive writer that creates detailed descriptions from prompts."
        }, {
          role: "user",
          content: enhancedPrompt
        }]
      });
      return response.data.choices[0].message?.content;
    } catch (error) {
      console.error('Error generating description:', error);
      throw error;
    }
  };

  // Step 3: Generate SVG from description
  const generateSVG = async (description: string) => {
    try {
      const response = await openai.createChatCompletion({
        model: "gpt-3.5-turbo",
        messages: [{
          role: "system",
          content: "You are an SVG generator that creates SVG code from descriptions."
        }, {
          role: "user",
          content: description
        }]
      });
      return response.data.choices[0].message?.content;
    } catch (error) {
      console.error('Error generating SVG:', error);
      throw error;
    }
  };

  // Handle the entire process
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      // Step 1: Enhance prompt
      const enhanced = await enhancePrompt(userInput);
      setEnhancedPrompt(enhanced || '');
      
      // Step 2: Generate description
      const desc = await generateDescription(enhanced || '');
      setDescription(desc || '');
      
      // Step 3: Generate SVG
      const svg = await generateSVG(desc || '');
      setSvgOutput(svg || '');
    } catch (error) {
      console.error('Error in chat composition:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-composition">
      <form onSubmit={handleSubmit}>
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Enter your prompt..."
          rows={4}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Generate'}
        </button>
      </form>

      {enhancedPrompt && (
        <div className="result-section">
          <h3>Enhanced Prompt:</h3>
          <pre>{enhancedPrompt}</pre>
        </div>
      )}

      {description && (
        <div className="result-section">
          <h3>Generated Description:</h3>
          <pre>{description}</pre>
        </div>
      )}

      {svgOutput && (
        <div className="result-section">
          <h3>Generated SVG:</h3>
          <div dangerouslySetInnerHTML={{ __html: svgOutput }} />
          <pre>{svgOutput}</pre>
        </div>
      )}
    </div>
  );
} 