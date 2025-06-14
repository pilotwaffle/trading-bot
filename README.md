# 🚀 Trading Bot Step-by-Step Start Guide

Follow these steps to set up and start your Alpaca paper trading bot using FastAPI and Docker Compose.

---

## 1. **Clone the Repository**

```bash
git clone https://github.com/pilotwaffle/trading-bot.git
cd trading-bot
```

---

## 2. **Prepare Your `.env` File**

- Copy and paste the following example, or edit your existing `.env` file with your [Alpaca Paper Trading](https://alpaca.markets/paper-dashboard/) API keys:

```bash
# .env
ALPACA_API_KEY=YOUR_ALPACA_PAPER_API_KEY
ALPACA_API_SECRET=YOUR_ALPACA_PAPER_API_SECRET

# Compatibility (recommended)
ALPACA_PAPER_API_KEY=YOUR_ALPACA_PAPER_API_KEY
ALPACA_PAPER_SECRET_KEY=YOUR_ALPACA_PAPER_API_SECRET
ALPACA_SECRET_KEY=YOUR_ALPACA_PAPER_API_SECRET

# Dashboard Authentication
APP_USER_ID=admin
APP_PASSWORD=admin123

# Optional settings
DEBUG=true
LOG_LEVEL=INFO
```

---

## 3. **(Optional) Review and Edit Configuration**

- You can adjust `APP_USER_ID`, `APP_PASSWORD`, or logging settings in `.env` as needed.

---

## 4. **Install Docker and Docker Compose**

- [Get Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/Mac, or use your package manager for Linux.
- Verify installation:

```bash
docker --version
docker compose version
```

---

## 5. **Build and Start the Bot with Docker Compose**

```bash
docker compose up --build
```
- This will:
  - Build the Docker image using your repo's `Dockerfile`
  - Load environment variables from `.env`
  - Expose FastAPI at [http://localhost:8000](http://localhost:8000)

---

## 6. **Access the Trading Bot Dashboard/API**

- Open your browser at: [http://localhost:8000](http://localhost:8000)
- The FastAPI docs are available at: [http://localhost:8000/docs](http://localhost:8000/docs)
- Login with credentials set in `.env` (`APP_USER_ID` / `APP_PASSWORD`).

---

## 7. **(Optional) View Logs and Data**

- Logs are written to `logs/` (mounted from the container).
- Trading data is stored in `data/` if your bot writes to this folder.

---

## 8. **Stopping the Bot**

- In your terminal, press `Ctrl+C` to stop.
- Or, to fully stop and clean up containers:

```bash
docker compose down
```

---

## 9. **Updating the Bot**

- Pull the latest code and rebuild:

```bash
git pull
docker compose up --build
```

---

## Troubleshooting

- **Check logs** in the `logs/` directory for errors.
- Make sure your `.env` has your correct Alpaca paper trading keys.
- For Docker issues, try restarting Docker Desktop or running with `sudo` on Linux.

---

## FAQ

**Q: Do I need to install Python packages manually?**  
A: No, Docker handles all requirements for you.

**Q: Can I run without Docker?**  
A: Yes! Just create a virtual environment, install `requirements.txt`, and run `python main.py`.  
But Docker is recommended for reproducibility.

**Q: Can I trade live?**  
A: Not by default—this setup is for Alpaca paper trading only. For live trading, you must update your `.env` and possibly code.

---

**That’s it! Your trading bot is ready to go.**

Happy (paper) trading! 🎉
