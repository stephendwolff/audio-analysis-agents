/**
 * Audio Analysis Frontend
 *
 * Demonstrates:
 * - File upload via REST API
 * - Streaming chat via WebSocket
 * - API key authentication
 */

// State
let currentTrackId = null;
let websocket = null;

// DOM Elements
const configForm = document.getElementById("config-form");
const apiUrlInput = document.getElementById("api-url");
const apiKeyInput = document.getElementById("api-key");
const uploadSection = document.getElementById("upload-section");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const chatSection = document.getElementById("chat-section");
const trackInfo = document.getElementById("track-info");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const connectionStatus = document.getElementById("connection-status");

// Initialize with config values
function initConfig() {
  apiUrlInput.value = CONFIG.API_BASE_URL;
  apiKeyInput.value = CONFIG.API_KEY;
}

// Save config
configForm.addEventListener("submit", (e) => {
  e.preventDefault();
  CONFIG.API_BASE_URL = apiUrlInput.value.replace(/\/$/, ""); // Remove trailing slash
  CONFIG.WS_BASE_URL = CONFIG.API_BASE_URL.replace(/^http/, "ws");
  CONFIG.API_KEY = apiKeyInput.value;
  showMessage("Configuration saved", "success");
});

// Upload file
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    showUploadStatus("Please select a file", "error");
    return;
  }

  showUploadStatus("Uploading...", "info");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/tracks/`, {
      method: "POST",
      headers: {
        "X-API-Key": CONFIG.API_KEY,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Upload failed");
    }

    const data = await response.json();
    currentTrackId = data.track_id;

    showUploadStatus(`Uploaded: ${data.filename}`, "success");
    showChatSection(data);

  } catch (error) {
    showUploadStatus(`Error: ${error.message}`, "error");
  }
});

// Show chat section and connect WebSocket
function showChatSection(trackData) {
  chatSection.style.display = "block";
  trackInfo.textContent = `Track: ${trackData.filename} (${trackData.track_id})`;
  chatMessages.innerHTML = "";

  connectWebSocket(trackData.track_id);
}

// Connect to WebSocket
function connectWebSocket(trackId) {
  if (websocket) {
    websocket.close();
  }

  const wsUrl = `${CONFIG.WS_BASE_URL}/ws/chat/${trackId}/?api_key=${encodeURIComponent(CONFIG.API_KEY)}`;
  setConnectionStatus("Connecting...", "info");

  websocket = new WebSocket(wsUrl);

  websocket.onopen = () => {
    setConnectionStatus("Connected", "success");
    questionInput.disabled = false;
  };

  websocket.onclose = (e) => {
    if (e.code === 4001) {
      setConnectionStatus("Unauthorized - check API key", "error");
    } else if (e.code === 4004) {
      setConnectionStatus("Track not found", "error");
    } else {
      setConnectionStatus("Disconnected", "error");
    }
    questionInput.disabled = true;
  };

  websocket.onerror = () => {
    setConnectionStatus("Connection error", "error");
  };

  websocket.onmessage = (event) => {
    handleWebSocketMessage(JSON.parse(event.data));
  };
}

// Handle incoming WebSocket messages
let currentResponseElement = null;

function handleWebSocketMessage(data) {
  switch (data.type) {
    case "tool_call":
      addSystemMessage(`Calling tool: ${data.tool}`);
      break;

    case "tool_result":
      const status = data.success ? "✓" : "✗";
      addSystemMessage(`Tool ${data.tool}: ${status}`);
      break;

    case "token":
      if (!currentResponseElement) {
        currentResponseElement = addAssistantMessage("");
      }
      currentResponseElement.textContent += data.text;
      scrollToBottom();
      break;

    case "done":
      currentResponseElement = null;
      break;

    case "error":
      addErrorMessage(data.message);
      currentResponseElement = null;
      break;
  }
}

// Send question
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question || !websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }

  addUserMessage(question);
  questionInput.value = "";

  websocket.send(JSON.stringify({
    type: "question",
    text: question,
  }));
});

// UI Helpers
function showUploadStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.className = `status ${type}`;
}

function setConnectionStatus(message, type) {
  connectionStatus.textContent = message;
  connectionStatus.className = `status ${type}`;
}

function showMessage(message, type) {
  // Could use a toast notification here
  console.log(`[${type}] ${message}`);
}

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addAssistantMessage(text) {
  const div = document.createElement("div");
  div.className = "message assistant";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function addSystemMessage(text) {
  const div = document.createElement("div");
  div.className = "message system";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addErrorMessage(text) {
  const div = document.createElement("div");
  div.className = "message error";
  div.textContent = `Error: ${text}`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Initialize
initConfig();
