/**
 * Configuration for the Audio Analysis Frontend
 *
 * These values can be changed without modifying the main app code.
 * In production, you might load these from environment variables
 * or a separate config endpoint.
 */

const CONFIG = {
  // API base URL (no trailing slash)
  API_BASE_URL: "http://localhost:8000",

  // API key for authentication
  API_KEY: "dev-api-key",

  // WebSocket base URL (ws:// or wss://)
  WS_BASE_URL: "ws://localhost:8000",
};

// Make config available globally
window.CONFIG = CONFIG;
