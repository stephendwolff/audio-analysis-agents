/**
 * Configuration for the Audio Analysis Frontend
 */

const CONFIG = {
  // API base URL (no trailing slash)
  API_BASE_URL: window.location.origin,

  // WebSocket base URL (ws:// or wss://)
  WS_BASE_URL: window.location.origin.replace(/^http/, "ws"),

  // JWT token (set after login/demo)
  ACCESS_TOKEN: null,
  REFRESH_TOKEN: null,
  IS_DEMO: false,
  REQUEST_LIMIT: 0,
};

// Make config available globally
window.CONFIG = CONFIG;
