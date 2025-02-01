import { Configuration } from 'openai';

export const getOpenAIConfig = () => {
  const apiKey = process.env.OPENAI_API_KEY;
  
  if (!apiKey) {
    throw new Error('OpenAI API key is not configured');
  }

  return new Configuration({
    apiKey: apiKey
  });
}; 