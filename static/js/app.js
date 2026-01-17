/**
 * Audio Analysis Frontend
 */

// State
let currentTrackId = null;
let websocket = null;
let requestCount = 0;
let trackStatus = "pending";

// Reconnection state
let reconnectAttempts = 0;
let reconnectTimeout = null;
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY = 1000; // 1 second

// DOM Elements
const authSection = document.getElementById("auth-section");
const authStatus = document.getElementById("auth-status");
const authButtons = document.getElementById("auth-buttons");
const demoBtn = document.getElementById("demo-btn");
const loginBtn = document.getElementById("login-btn");
const loginForm = document.getElementById("login-form");
const cancelLoginBtn = document.getElementById("cancel-login");
const uploadSection = document.getElementById("upload-section");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const chatSection = document.getElementById("chat-section");
const trackInfo = document.getElementById("track-info");
const demoInfo = document.getElementById("demo-info");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const connectionStatus = document.getElementById("connection-status");

// Auth handlers
demoBtn.addEventListener("click", async () => {
  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/auth/demo/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    if (!response.ok) throw new Error("Failed to start demo");

    const data = await response.json();
    CONFIG.ACCESS_TOKEN = data.access;
    CONFIG.REFRESH_TOKEN = data.refresh;
    CONFIG.IS_DEMO = true;
    CONFIG.REQUEST_LIMIT = data.request_limit;
    requestCount = 0;

    showAuthenticatedUI();
  } catch (error) {
    authStatus.textContent = `Error: ${error.message}`;
    authStatus.className = "status error";
  }
});

loginBtn.addEventListener("click", () => {
  authButtons.style.display = "none";
  loginForm.style.display = "block";
});

cancelLoginBtn.addEventListener("click", () => {
  loginForm.style.display = "none";
  authButtons.style.display = "block";
});

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/auth/login/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Login failed");
    }

    const data = await response.json();
    CONFIG.ACCESS_TOKEN = data.access;
    CONFIG.REFRESH_TOKEN = data.refresh;
    CONFIG.IS_DEMO = data.is_demo;

    showAuthenticatedUI();
  } catch (error) {
    authStatus.textContent = `Error: ${error.message}`;
    authStatus.className = "status error";
  }
});

function showAuthenticatedUI() {
  authButtons.style.display = "none";
  loginForm.style.display = "none";

  if (CONFIG.IS_DEMO) {
    authStatus.textContent = `Demo mode: ${CONFIG.REQUEST_LIMIT} requests available`;
  } else {
    authStatus.textContent = "Logged in";
  }
  authStatus.className = "status success";

  uploadSection.style.display = "block";
}

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
        Authorization: `Bearer ${CONFIG.ACCESS_TOKEN}`,
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
  trackInfo.textContent = `Track: ${trackData.filename}`;
  chatMessages.innerHTML = "";
  updateDemoInfo();

  // Reset track status
  trackStatus = trackData.status || "pending";
  updateTrackStatus(trackStatus, "Connecting...");

  connectWebSocket(trackData.track_id);
}

function updateDemoInfo() {
  if (CONFIG.IS_DEMO) {
    const remaining = CONFIG.REQUEST_LIMIT - requestCount;
    demoInfo.textContent = `Demo: ${remaining} requests remaining`;
    demoInfo.className = remaining <= 1 ? "status error" : "status info";
  } else {
    demoInfo.textContent = "";
  }
}

// Update track status display and input state
function updateTrackStatus(status, message) {
  trackStatus = status;

  // Update track info with status
  const statusText = message || getStatusMessage(status);
  trackInfo.textContent = statusText;

  // Disable input unless track is ready AND connected
  const isReady = status === "ready";
  const isConnected = websocket && websocket.readyState === WebSocket.OPEN;
  questionInput.disabled = !(isReady && isConnected);

  // Update placeholder text
  if (!isReady) {
    questionInput.placeholder = "Waiting for analysis to complete...";
  } else if (!isConnected) {
    questionInput.placeholder = "Connecting...";
  } else {
    questionInput.placeholder = "Ask a question about your audio...";
  }
}

function getStatusMessage(status) {
  const messages = {
    pending: "Waiting to start analysis...",
    analyzing: "Analyzing audio...",
    ready: "Ready for questions",
    failed: "Analysis failed",
  };
  return messages[status] || status;
}

// Connect to WebSocket
function connectWebSocket(trackId, isReconnect = false) {
  // Clear any pending reconnect
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }

  if (websocket) {
    websocket.close();
  }

  // Store trackId for reconnection
  currentTrackId = trackId;

  const wsUrl = `${CONFIG.WS_BASE_URL}/ws/chat/${trackId}/?token=${encodeURIComponent(CONFIG.ACCESS_TOKEN)}`;
  setConnectionStatus(isReconnect ? `Reconnecting (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...` : "Connecting...", "info");

  websocket = new WebSocket(wsUrl);

  websocket.onopen = () => {
    setConnectionStatus("Connected", "success");
    reconnectAttempts = 0; // Reset on successful connection

    // Update input state based on track status
    updateTrackStatus(trackStatus, null);
  };

  websocket.onclose = (e) => {
    questionInput.disabled = true;

    // Don't reconnect for auth/not-found errors
    if (e.code === 4001) {
      setConnectionStatus("Unauthorized", "error");
      reconnectAttempts = MAX_RECONNECT_ATTEMPTS; // Prevent reconnection
      return;
    }
    if (e.code === 4004) {
      setConnectionStatus("Track not found", "error");
      reconnectAttempts = MAX_RECONNECT_ATTEMPTS; // Prevent reconnection
      return;
    }

    // Attempt reconnection for other disconnects
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS && currentTrackId) {
      scheduleReconnect(currentTrackId);
    } else {
      setConnectionStatus("Disconnected", "error");
    }
  };

  websocket.onerror = () => {
    // onerror is always followed by onclose, so reconnection is handled there
    setConnectionStatus("Connection error", "error");
  };

  websocket.onmessage = (event) => {
    handleWebSocketMessage(JSON.parse(event.data));
  };
}

function scheduleReconnect(trackId) {
  reconnectAttempts++;
  const delay = BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttempts - 1); // Exponential backoff
  setConnectionStatus(`Reconnecting in ${delay / 1000}s...`, "info");

  reconnectTimeout = setTimeout(() => {
    connectWebSocket(trackId, true);
  }, delay);
}

// Handle incoming WebSocket messages
let currentResponseElement = null;

function handleWebSocketMessage(data) {
  switch (data.type) {
    case "status":
      // Track status update from server
      updateTrackStatus(data.status, data.message);
      if (data.status === "analyzing" || data.status === "pending") {
        addSystemMessage(data.message);
      } else if (data.status === "ready") {
        addSystemMessage("Analysis complete. You can now ask questions.");
      } else if (data.status === "failed") {
        addErrorMessage(`Analysis failed: ${data.message}`);
      }
      break;

    case "thinking":
      // Analysis progress from Celery worker
      addThinkingMessage(data.message);
      break;

    case "tool_call":
      addSystemMessage(`Calling tool: ${data.tool}`);
      break;

    case "tool_result":
      const status = data.success ? "+" : "x";
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

  // Check if track is ready
  if (trackStatus !== "ready") {
    addErrorMessage("Please wait for analysis to complete before asking questions.");
    return;
  }

  // Check demo limit
  if (CONFIG.IS_DEMO && requestCount >= CONFIG.REQUEST_LIMIT) {
    addErrorMessage("Demo request limit reached. Please login for unlimited access.");
    return;
  }

  addUserMessage(question);
  questionInput.value = "";

  if (CONFIG.IS_DEMO) {
    requestCount++;
    updateDemoInfo();
  }

  websocket.send(
    JSON.stringify({
      type: "question",
      text: question,
    })
  );
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

function addThinkingMessage(text) {
  const div = document.createElement("div");
  div.className = "message thinking";
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
