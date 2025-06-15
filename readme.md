# 🚀 Industrial Crypto Trading Bot & Dashboard — Step-by-Step Startup Guide

This guide combines setup steps for an industrial-grade crypto trading bot and dashboard using FastAPI (backend) and a SPA dashboard frontend. Both Docker Compose and manual Python/venv setups are covered.

---

## 1. **Clone the Repository**

```bash
git clone https://github.com/pilotwaffle/trading-bot.git
cd trading-bot
```

---

## 2. **Set up your Environment**

### a. (Recommended) Create a Virtual Environment

```sh
python -m venv venv
```
Activate it:
- **Windows:**  
  `venv\Scripts\activate`
- **Mac/Linux:**  
  `source venv/bin/activate`

### b. Install All Dependencies

```sh
pip install -r requirements.txt
```

*Alternatively, you can use Docker (see section 5 below for details).*

---

## 3. **Configure Environment Variables**

Create a `.env` file in your project root with the following example (edit with your real keys):

```env
# Alpaca API keys (Paper Trading)
ALPACA_API_KEY=YOUR_ALPACA_PAPER_API_KEY
ALPACA_API_SECRET=YOUR_ALPACA_PAPER_API_SECRET

# Compatibility (optional/recommended)
ALPACA_PAPER_API_KEY=YOUR_ALPACA_PAPER_API_KEY
ALPACA_PAPER_SECRET_KEY=YOUR_ALPACA_PAPER_API_SECRET
ALPACA_SECRET_KEY=YOUR_ALPACA_PAPER_API_SECRET

# CoinMarketCap API (if using crypto features)
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key_here

# Alternate Alpaca key names for some clients
APCA_API_KEY_ID=your_alpaca_api_key_id_here
APCA_API_SECRET_KEY=your_alpaca_api_secret_key_here

# Dashboard Authentication
APP_USER_ID=admin
APP_PASSWORD=admin123

# Optional settings
DEBUG=true
LOG_LEVEL=INFO
```

---

## 4. **Prepare Backend & Frontend Files**

Ensure your directories contain:

**Backend (`app/`):**
- `main.py` (FastAPI app with all routers and startup logic)
- `api_extra_routes.py` (e.g., `/api/coinmarketcap-prices`, `/api/alpaca-status`)
- `trading_engine.py` (your trading logic)

**Frontend (`static/`):**
- `/static/dashboard.html`
- `/static/css/dashboard.css`
- `/static/js/dashboard.js`

---

## 5. **Start the Bot**

### Option A: **With Docker (Recommended for reproducibility)**

Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed (or Docker Engine on Linux).

Build and start:

```bash
docker compose up --build
```
- This builds the Docker image, loads `.env` variables, and exposes FastAPI at [http://localhost:8000](http://localhost:8000).
- Logs are written to `logs/`.
- Trading data (if used) is written to `data/`.

To stop and clean up:

```bash
docker compose down
```

### Option B: **Manual (venv + Uvicorn)**

Make sure your virtual environment is active.

Start the server:

```sh
uvicorn app.main:app --reload
```
If you see `'uvicorn' is not recognized...`:
```sh
pip install uvicorn[standard]
```
Or run:
```sh
python -m uvicorn app.main:app --reload
```

---

## 6. **Access the Trading Bot Dashboard/API**

Open your browser at: [http://localhost:8000](http://localhost:8000)

- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Login with credentials from `.env` (`APP_USER_ID` / `APP_PASSWORD`).
- Dashboard tabs include: System Status, Crypto Prices, Trading, Strategies, Analytics, Chat, ML Training, etc.

---

## 7. **Bot & Dashboard Operation**

- **Bot startup:** Trading engine starts with FastAPI.
- **System Status Tab:** Check Alpaca API status.
- **Crypto Prices Tab:** Live crypto data.
- **Trade, strategies, analytics, chat, ML training:** Use dashboard tabs as needed.

---

## 8. **Stopping the Bot**

- **Docker:** Press `Ctrl+C` or use `docker compose down`.
- **Manual:** Stop the `uvicorn` process with `Ctrl+C` in your terminal.

---

## 9. **Updating the Bot**

```bash
git pull
docker compose up --build  # (if using Docker)
```

---

## 10. **Troubleshooting**

- **Missing dependencies:** Run `pip install -r requirements.txt`
- **API key errors:** Check your `.env` file.
- **File not found:** Ensure your directory structure matches this guide.
- **Uvicorn not found:** Run `pip install uvicorn[standard]`
- **Port conflict:** Use `--port 8080` or another port with Uvicorn.
- **Docker issues:** Restart Docker Desktop or use `sudo` on Linux.
- **Logs:** Check `logs/` directory.

---

## 11. **Production Deployment**

- Use a process manager (e.g., `gunicorn`, `supervisor`).
- Run Uvicorn with `--host 0.0.0.0 --port 80` for production.
- Use HTTPS and restrict CORS for security.

---

## **FAQ**

**Q: Do I need to install Python packages manually?**  
A: No, Docker handles all requirements for you. If not using Docker, install with `pip install -r requirements.txt`.

**Q: Can I run without Docker?**  
A: Yes! Use a Python virtual environment as above.

**Q: Can I trade live?**  
A: Not by default—this setup is for Alpaca paper trading only. For live trading, update your `.env` and code as needed.

**Q: Database setup?**  
A: If your bot uses a database, run any migration/setup scripts as required by your backend.

---

**You’re ready!**  
If you need help with any step or get an error, copy the error message here for help.

Happy (paper) trading! 🎉