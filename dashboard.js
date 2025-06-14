// ========== Helper Functions ==========

async function apiCall(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let msg = `API error: ${response.status}`;
    try { msg = (await response.json()).message || msg; } catch {}
    throw new Error(msg);
  }
  return response.json();
}

function showAlert(message, type = "info") {
  const alertBox = document.getElementById('main-alert');
  alertBox.textContent = message;
  alertBox.className = `alert alert-${type}`;
  alertBox.style.display = 'block';
  setTimeout(() => { alertBox.style.display = 'none'; }, 4000);
}

function formatCurrency(value) {
  return value !== undefined && value !== null
    ? '$' + Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : '';
}
function formatPercentage(value) {
  return value !== undefined && value !== null
    ? (value * 100).toFixed(2) + '%'
    : '';
}

// ========== Trading Page Functions ==========
async function loadTradingData() {
  await loadMarketData();
  await populateSymbolDropdown();
}

async function populateSymbolDropdown() {
  try {
    const data = await apiCall('/api/market-data');
    const select = document.getElementById('trade-symbol');
    select.innerHTML = '<option value="">Select a cryptocurrency</option>';
    Object.keys(data.market_data).forEach(symbol => {
      const option = document.createElement('option');
      option.value = symbol;
      option.textContent = symbol;
      select.appendChild(option);
    });
  } catch (error) {
    showAlert('Failed to populate symbols: ' + error.message, 'danger');
  }
}

async function executeTrade(event) {
  event.preventDefault();
  const formData = new FormData(event.target);
  const tradeData = {
    symbol: formData.get('symbol'),
    side: formData.get('side'),
    amount: parseFloat(formData.get('amount')),
    price: formData.get('price') ? parseFloat(formData.get('price')) : null
  };
  try {
    const result = await apiCall('/api/trade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tradeData)
    });
    showAlert(result.message, 'success');
    event.target.reset();
    setTimeout(() => {
      loadPositions();
      loadPerformance();
    }, 1000);
  } catch (error) {
    showAlert(`Trade failed: ${error.message}`, 'danger');
  }
}

// ========== Positions Page Functions ==========
async function loadPositions() {
  try {
    const data = await apiCall('/api/positions');
    const container = document.getElementById('positions-content');
    if (data.positions.length === 0) {
      container.innerHTML = '<div>No open positions</div>';
      return;
    }
    const tableRows = data.positions.map(position => `
      <tr>
        <td>${position.symbol}</td>
        <td>${position.amount.toFixed(6)}</td>
        <td>${formatCurrency(position.entry_price)}</td>
        <td>${formatCurrency(position.current_price)}</td>
        <td>${formatCurrency(position.unrealized_pnl)}</td>
        <td>${formatPercentage(position.percentage_change)}</td>
        <td><button class="btn btn-sm btn-danger" onclick="closePosition('${position.symbol}')">Close</button></td>
      </tr>
    `).join('');
    const table = `
      <table class="table table-sm">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Amount</th>
            <th>Entry Price</th>
            <th>Current Price</th>
            <th>P&L</th>
            <th>Change %</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    `;
    container.innerHTML = table;
  } catch (error) {
    showAlert('Failed to load positions: ' + error.message, 'danger');
    document.getElementById('positions-content').innerHTML = '<div>Error loading positions</div>';
  }
}

async function closePosition(symbol) {
  if (!confirm(`Are you sure you want to close position for ${symbol}?`)) return;
  try {
    await apiCall('/api/trade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: symbol, side: 'sell', amount: 999999 })
    });
    showAlert(`Position closed: ${symbol}`, 'success');
    loadPositions();
    loadPerformance();
  } catch (error) {
    showAlert(`Failed to close position: ${error.message}`, 'danger');
  }
}

// ========== Crypto Market Page Functions ==========
async function loadCryptoMarket() {
  try {
    const data = await apiCall('/api/crypto-rankings');
    const container = document.getElementById('crypto-grid');
    if (!data.rankings || data.rankings.length === 0) {
      container.innerHTML = '<div>No market data available</div>';
      return;
    }
    const cryptoCards = data.rankings.map((crypto, index) => `
      <div class="crypto-card">
        <div class="crypto-rank">#${index + 1}</div>
        <div class="crypto-name">${crypto.name}</div>
        <div class="crypto-price">${formatCurrency(crypto.price)}</div>
        <div class="crypto-change">${formatPercentage(crypto.change_24h)}</div>
        <div class="crypto-volume">Vol: ${crypto.volume.toLocaleString()}</div>
      </div>
    `).join('');
    container.innerHTML = cryptoCards;
  } catch (error) {
    showAlert('Failed to load crypto market: ' + error.message, 'danger');
    document.getElementById('crypto-grid').innerHTML = '<div>Error loading market data</div>';
  }
}

// ========== Strategies Page Functions ==========
async function loadStrategies() {
  try {
    const data = await apiCall('/api/strategies');
    const container = document.getElementById('strategies-content');
    const strategiesHtml = data.strategies.map(strategy => `
      <div class="strategy-card">
        <div class="strategy-header">
          <strong>${strategy.name} Strategy</strong>
          <button class="btn btn-sm ${strategy.enabled ? 'btn-danger' : 'btn-success'}" onclick="toggleStrategy('${strategy.name}')">
            ${strategy.enabled ? 'Disable' : 'Enable'}
          </button>
        </div>
        <div class="strategy-params">
          <pre>${JSON.stringify(strategy.params, null, 2)}</pre>
        </div>
        <div class="strategy-status">
          <span class="status-dot ${strategy.enabled ? 'status-active' : 'status-inactive'}"></span>
          <span>${strategy.enabled ? 'Active' : 'Inactive'}</span>
        </div>
      </div>
    `).join('');
    container.innerHTML = strategiesHtml;
  } catch (error) {
    showAlert('Failed to load strategies: ' + error.message, 'danger');
    document.getElementById('strategies-content').innerHTML = '<div>Error loading strategies</div>';
  }
}

async function toggleStrategy(strategyName) {
  try {
    const result = await apiCall(`/api/strategies/${strategyName}/toggle`, { method: 'POST' });
    showAlert(`Strategy ${strategyName} ${result.enabled ? 'enabled' : 'disabled'}`, 'success');
    loadStrategies();
  } catch (error) {
    showAlert(`Failed to toggle strategy: ${error.message}`, 'danger');
  }
}

// ========== ML Training Functions ==========
async function loadMLStatus() {
  try {
    const data = await apiCall('/api/ml/status');
    document.getElementById('model-status').textContent = data.is_trained ? 'Trained' : 'Not Trained';
    document.getElementById('last-training').textContent = data.last_training || 'Never';
    document.getElementById('model-accuracy').textContent = data.accuracy ? `${(data.accuracy * 100).toFixed(2)}%` : 'N/A';
  } catch (error) {
    showAlert('Failed to load ML status: ' + error.message, 'danger');
  }
}

async function trainMLModel(event) {
  event.preventDefault();
  const formData = new FormData(event.target);
  const requestData = {
    days: parseInt(formData.get('days')),
    symbols: []
  };
  try {
    const result = await apiCall('/api/ml/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });
    showAlert(result.message, 'success');
    document.getElementById('training-progress').style.display = 'block';
    simulateTrainingProgress();
  } catch (error) {
    showAlert(`Training failed: ${error.message}`, 'danger');
  }
}

function simulateTrainingProgress() {
  const progressFill = document.getElementById('progress-fill');
  const statusText = document.getElementById('training-status');
  let progress = 0;
  const steps = [
    'Preparing training data...',
    'Loading historical data...',
    'Feature engineering...',
    'Training model...',
    'Validating model...',
    'Training complete!'
  ];
  const interval = setInterval(() => {
    progress += Math.random() * 20;
    if (progress > 100) progress = 100;
    progressFill.style.width = `${progress}%`;
    statusText.textContent = steps[Math.floor((progress / 100) * (steps.length - 1))];
    if (progress >= 100) {
      clearInterval(interval);
      setTimeout(() => {
        document.getElementById('training-progress').style.display = 'none';
        loadMLStatus();
      }, 2000);
    }
  }, 1000);
}

// ========== Chat Functions ==========
function handleChatKeyPress(event) {
  if (event.key === 'Enter') {
    sendChatMessage();
  }
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;
  addChatMessage('You', message);
  input.value = '';
  try {
    const result = await apiCall('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    });
    addChatMessage('Bot', result.response);
  } catch (error) {
    addChatMessage('System', `Error: ${error.message}`);
  }
}

function addChatMessage(user, message) {
  const messagesContainer = document.getElementById('chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = 'chat-message';
  messageDiv.innerHTML = `
    <div class="chat-user">${user}</div>
    <div class="chat-text">${message}</div>
    <div class="chat-time">${new Date().toLocaleTimeString()}</div>
  `;
  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateChatMessages(data) {
  addChatMessage(data.user, data.message);
  if (data.response) {
    addChatMessage('Bot', data.response);
  }
}

// ========== Analytics Functions ==========
async function loadAnalytics() {
  try {
    const data = await apiCall('/api/analytics/summary');
    const container = document.getElementById('analytics-summary');
    const analyticsHtml = `
      <div class="analytics-card"><div>Total Trades</div><div>${data.total_trades}</div></div>
      <div class="analytics-card"><div>Winning Trades</div><div>${data.winning_trades}</div></div>
      <div class="analytics-card"><div>Win Rate</div><div>${data.win_rate.toFixed(1)}%</div></div>
      <div class="analytics-card"><div>Total P&L</div><div>${formatCurrency(data.total_pnl)}</div></div>
      <div class="analytics-card"><div>Average P&L</div><div>${formatCurrency(data.average_pnl)}</div></div>
      <div class="analytics-card"><div>Best Trade</div><div>${formatCurrency(data.best_trade)}</div></div>
      <div class="analytics-card"><div>Worst Trade</div><div>${formatCurrency(data.worst_trade)}</div></div>
    `;
    container.innerHTML = analyticsHtml;
  } catch (error) {
    showAlert('Failed to load analytics: ' + error.message, 'danger');
    document.getElementById('analytics-summary').innerHTML = '<div>Error loading analytics</div>';
  }
}

// ========== Bot Control Functions ==========
async function toggleBot() {
  const button = document.getElementById('toggle-bot');
  const isRunning = button.textContent.trim() === 'Stop Bot';
  try {
    const endpoint = isRunning ? '/api/bot/stop' : '/api/bot/start';
    const result = await apiCall(endpoint, { method: 'POST' });
    showAlert(result.message, 'success');
    updateBotStatus(!isRunning);
  } catch (error) {
    showAlert(`Failed to ${isRunning ? 'stop' : 'start'} bot: ${error.message}`, 'danger');
  }
}

function updateBotStatus(running) {
  const statusElement = document.getElementById('bot-status');
  const buttonElement = document.getElementById('toggle-bot');
  if (running) {
    statusElement.className = 'status-indicator status-running';
    statusElement.innerHTML = '● Running';
    buttonElement.textContent = 'Stop Bot';
    buttonElement.className = 'btn btn-danger';
  } else {
    statusElement.className = 'status-indicator status-stopped';
    statusElement.innerHTML = '● Stopped';
    buttonElement.textContent = 'Start Bot';
    buttonElement.className = 'btn btn-success';
  }
}

async function syncData() {
  try {
    showAlert('Syncing data...', 'info');
    loadTradingData();
    loadPositions();
    loadCryptoMarket();
    loadStrategies();
    loadMLStatus();
    loadAnalytics();
  } catch (error) {
    showAlert('Sync failed: ' + error.message, 'danger');
  }
}

// ========== Market Data Functions ==========
async function loadMarketData() {
  try {
    const data = await apiCall('/api/market-data');
    updateMarketData(data.market_data);
  } catch (error) {
    showAlert('Failed to load market data: ' + error.message, 'danger');
  }
}
function updateMarketData(marketData) {
  // Optionally implement to show summary market data
}

// ========== Performance/Status Loaders ==========
async function loadPerformance() {
  // Implement if you have a performance endpoint
}
async function loadBotStatus() {
  // Implement if you have a bot status endpoint
}