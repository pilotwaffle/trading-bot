// ========== Tab Navigation ==========
function showTab(tab) {
  document.querySelectorAll('.tab-content').forEach(
    el => el.classList.remove('active')
  );
  document.querySelectorAll('.tab-btn').forEach(
    el => el.classList.remove('active')
  );
  document.getElementById('tab-' + tab).classList.add('active');
  Array.from(document.getElementsByClassName('tab-btn'))
    .find(btn => btn.getAttribute('onclick').includes(tab))
    .classList.add('active');
  if (tab === 'crypto-prices') loadCryptoPrices();
  if (tab === 'system-status') checkAlpacaStatus();
  if (tab === 'ml-train') loadMLStatus();
}

// ========== Helper Functions ==========
// [Keep all your original helpers here, unchanged...]

// ========== ML Training Tab: Add event and progress logic ==========
document.addEventListener('DOMContentLoaded', () => {
  loadTradingData();
  loadPositions();
  loadCryptoMarket();
  loadStrategies();
  loadMLStatus();
  loadAnalytics();

  // ML Training tab events
  document.getElementById('training-form').onsubmit = function(e) {
    e.preventDefault();
    const statusText = document.getElementById('training-status-text');
    const progressBar = document.getElementById('training-progress-bar');
    statusText.textContent = 'Training...';
    progressBar.style.width = '0%';
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress > 100) progress = 100;
      progressBar.style.width = progress + '%';
      if (progress >= 100) {
        clearInterval(interval);
        statusText.textContent = 'Training Complete!';
        setTimeout(loadMLStatus, 1000);
      }
    }, 700);
  };
});

// ========== Crypto Prices Tab: CoinGecko/CoinMarketCap ==========
async function loadCryptoPrices() {
  const provider = document.getElementById('price-provider')?.value || 'coingecko';
  const tbody = document.querySelector('#crypto-table tbody');
  tbody.innerHTML = '<tr><td colspan="4">Loading...</td></tr>';
  let url;
  if (provider === 'coingecko') {
    url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false';
  } else {
    url = '/api/coinmarketcap-prices'; // Proxy endpoint, must be implemented in backend.
  }
  try {
    const res = await fetch(url);
    const data = await res.json();
    tbody.innerHTML = '';
    (data || []).forEach(coin => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${coin.symbol ? coin.symbol.toUpperCase() : ''}</td>
        <td>${coin.name || ''}</td>
        <td>$${coin.current_price ? coin.current_price.toLocaleString() : coin.price?.toLocaleString() || ''}</td>
        <td style="color:${(coin.price_change_percentage_24h ?? coin.change_24h) >= 0 ? '#00d084' : '#ff4d4f'}">
          ${(coin.price_change_percentage_24h ?? coin.change_24h ?? 0).toFixed(2)}%
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="4">Error loading prices</td></tr>';
  }
}

// ========== System Status Tab: Alpaca connection ==========
async function checkAlpacaStatus() {
  const statusDot = document.getElementById('alpaca-status');
  const statusText = document.getElementById('alpaca-status-text');
  const accountStatus = document.getElementById('alpaca-account-status');
  statusDot.className = 'status-dot status-offline';
  statusText.textContent = 'Checking...';
  accountStatus.textContent = '';
  try {
    const res = await fetch('/api/alpaca-status');
    if (res.ok) {
      const data = await res.json();
      statusDot.className = 'status-dot status-online';
      statusText.textContent = 'Connected';
      accountStatus.textContent = data.account_status ? `Account Status: ${data.account_status}` : '';
    } else {
      statusText.textContent = 'Not Connected';
    }
  } catch {
    statusText.textContent = 'Error';
  }
}

// ========== All your original functions BELOW here ==========
// (copy/paste all your original dashboard.js functions below this comment, unchanged)
// ...