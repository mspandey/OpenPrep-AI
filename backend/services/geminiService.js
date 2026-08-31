const { GoogleGenerativeAI } = require('@google/generative-ai');
const NodeCache = require('node-cache');
const crypto = require('crypto');
const { splitIntoChunks } = require('../utils/textChunking');
const { toLocalDateString } = require('../utils/dateUtils');
const CircuitBreaker = require('./circuitBreaker');
const AIContractVersioningService = require('./aiContractVersioningService');
const AIGenerationCacheService = require('./aiGenerationCacheService');

// Notes larger than this are split into semantic chunks across multiple Gemini passes
const NOTE_SUMMARY_CHUNK_MAX_CHARS = 11000;
const NOTE_DIGEST_MAX_CHARS = 5000;

// Initialize Gemini API client with validation
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  throw new Error('CRITICAL: GEMINI_API_KEY is not defined in the environment variables.');
}

const aiClient = new GoogleGenerativeAI(apiKey);
const cache = new NodeCache({ stdTTL: 3600, checkperiod: 120 });
const circuitBreaker = new CircuitBreaker({ threshold: 5, timeout: 60000 });

class GeminiRateLimitError extends Error {
  constructor(message) {
    super(message);
    this.name = 'GeminiRateLimitError';
    this.status = 429;
  }
}

class GeminiServerError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = 'GeminiServerError';
    this.status = status;
  }
}

/**
 * Generates a collision-resistant SHA-256 hash for caching prompts and configs.
 */
function hashKey(data) {
  return crypto.createHash('sha256').update(JSON.stringify(data)).digest('hex');
}

/**
 * Executes an AI generation call with exponential backoff retry logic and circuit breaker protection.
 */
async function generateWithRetry(modelName, promptPayload, options = {}) {
  const cacheKey = hashKey({ modelName, promptPayload, options });
  const cachedResult = cache.get(cacheKey);
  if (cachedResult) {
    return cachedResult;
  }

  const model = aiClient.getGenerativeModel({ model: modelName, ...options });
  
  const executeCall = async () => {
    let attempts = 0;
    const maxRetries = 3;
    let delay = 1000;

    while (attempts < maxRetries) {
      try {
        const result = await model.generateContent(promptPayload);
        const responseText = result.response.text();
        cache.set(cacheKey, responseText);
        return responseText;
      } catch (error) {
        attempts++;
        const status = error.status || error.statusCode;
        const isRateLimit = status === 429 || error.message?.includes('RESOURCE_EXHAUSTED');
        const isServerError = status >= 500 || error.message?.includes('INTERNAL');

        if ((isRateLimit || isServerError) && attempts < maxRetries) {
          const jitter = Math.random() * 200;
          await new Promise((resolve) => setTimeout(resolve, delay + jitter));
          delay *= 2; // Exponential backoff
          continue;
        }

        if (isRateLimit) {
          throw new GeminiRateLimitError('Gemini API rate limit exceeded. Please try again later.');
        }
        throw new GeminiServerError(error.message, status || 500);
      }
    }
  };

  return circuitBreaker.fire(executeCall);
}

module.exports = {
  aiClient,
  generateWithRetry,
  hashKey,
  GeminiRateLimitError,
  GeminiServerError,
  NOTE_SUMMARY_CHUNK_MAX_CHARS,
  NOTE_DIGEST_MAX_CHARS
};
