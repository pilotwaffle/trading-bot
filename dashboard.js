// --- Tab Navigation ---
function showTab(tab) {
    document.querySelectorAll('.tab-content').forEach(
        el => el.classList.remove('active')
    );
    document.querySelectorAll('.tab-btn').forEach(
        el => el.classList.remove('active')
    );
    const tabSection = document.getElementById('tab-' + tab);
    if (tabSection) tabSection.classList.add('active');

    const btn = Array.from(document.getElementsByClassName('tab-btn'))
        .find(btn => btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tab));
    if (btn) btn.classList.add('active');
}

// --- Simulated Training Logic ---
function initTrainingForm() {
    const trainingForm = document.getElementById('training-form');
    if (trainingForm) {
        trainingForm.onsubmit = function(e) {
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
                }
            }, 700);
        };
    }
}

// --- Populate Crypto Dropdown for Trading ---
async function loadCryptoMarket() {
    const select = document.getElementById('trade-symbol');
    if (!select) return;
    select.innerHTML = '<option value="">Loading...</option>';
    try {
        // Try backend first
        let coins;
        try {
            const res = await fetch('/api/market-data');
            const data = await res.json();
            coins = data.coins || data || [];
        } catch {
            // Fallback to CoinGecko
            const url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false';
            const res = await fetch(url);
            coins = await res.json();
        }
        select.innerHTML = '';
        coins.forEach(coin => {
            const option = document.createElement('option');
            option.value = coin.symbol.toUpperCase();
            option.text = `${coin.symbol.toUpperCase()} - ${coin.name}`;
            select.appendChild(option);
        });
    } catch (e) {
        // Fallback in case of total error
        select.innerHTML = `
            <option value="BTC">BTC - Bitcoin</option>
            <option value="ETH">ETH - Ethereum</option>
            <option value="SOL">SOL - Solana</option>
            <option value="DOGE">DOGE - Dogecoin</option>
        `;
    }
}

// --- Fetch Crypto Prices from CoinGecko or CoinMarketCap proxy ---
async function loadCryptoPrices() {
    const provider = document.getElementById('price-provider')?.value || 'coingecko';
    const tbody = document.querySelector('#crypto-table tbody');
    tbody.innerHTML = '<tr><td colspan="4">Loading...</td></tr>';
    let url;
    if (provider === 'coingecko') {
        url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false';
    } else {
        url = '/api/coinmarketcap-prices'; // backend proxy
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

// --- Check Alpaca API Connection (via backend proxy) ---
async function checkAlpacaStatus() {
    const statusDot = document.getElementById('alpaca-status');
    const statusText = document.getElementById('alpaca-status-text');
    const accountStatus = document.getElementById('alpaca-account-status');
    statusDot.className = 'status-dot status-offline';
    statusText.textContent = 'Checking...';
    if (accountStatus) accountStatus.textContent = '';
    try {
        // Call your own backend, which proxies to Alpaca and protects API keys
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

// --- Toggle Bot Start/Stop ---
function toggleBot() {
    const botStatus = document.getElementById('bot-status');
    const toggleBtn = document.getElementById('toggle-bot');
    if (botStatus && toggleBtn) {
        if (botStatus.classList.contains('status-stopped')) {
            botStatus.classList.remove('status-stopped');
            botStatus.classList.add('status-running');
            botStatus.textContent = '● Running';
            toggleBtn.textContent = 'Stop Bot';
            toggleBtn.classList.remove('btn-success');
            toggleBtn.classList.add('btn-danger');
        } else {
            botStatus.classList.remove('status-running');
            botStatus.classList.add('status-stopped');
            botStatus.textContent = '● Stopped';
            toggleBtn.textContent = 'Start Bot';
            toggleBtn.classList.remove('btn-danger');
            toggleBtn.classList.add('btn-success');
        }
    }
}

// --- Dummy Functions for Demo Buttons ---
function syncData() {
    const mainAlert = document.getElementById('main-alert');
    if (mainAlert) {
        mainAlert.textContent = 'Data synced!';
        mainAlert.style.display = 'block';
        setTimeout(() => {
            mainAlert.style.display = 'none';
        }, 1500);
    }
}

// --- Dummy Functions for Chat Section ---
function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');
    if (input && input.value.trim() && messages) {
        const msg = document.createElement('div');
        msg.textContent = "You: " + input.value;
        messages.appendChild(msg);
        input.value = '';
        messages.scrollTop = messages.scrollHeight;
    }
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

// --- Ensure dropdown is filled and handlers are set on page load ---
document.addEventListener('DOMContentLoaded', () => {
    loadCryptoMarket();
    initTrainingForm();

    // Tab Button Event Handlers
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const text = this.textContent.trim().toLowerCase();
            if (text === 'dashboard') showTab('dashboard');
            if (text === 'train bot') showTab('ml-train');
            if (text === 'crypto prices') {
                showTab('crypto-prices');
                loadCryptoPrices();
            }
            if (text === 'system status') {
                showTab('system-status');
                checkAlpacaStatus();
            }
        });
    });
});