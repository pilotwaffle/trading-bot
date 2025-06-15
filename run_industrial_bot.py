#!/usr/bin/env python3
"""
Industrial Crypto Trading Bot Startup Script
Complete setup and launch for production-ready trading bot
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🤖 INDUSTRIAL CRYPTO TRADING BOT v2.0                    ║
║                                                              ║
║    ▶ Real-time cryptocurrency trading                        ║
║    ▶ Machine learning predictions                            ║
║    ▶ Automated strategy execution                            ║
║    ▶ Industrial-grade monitoring                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_system_requirements():
    """Check system requirements and dependencies"""
    print("🔍 Checking system requirements...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Exiting.")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected.")

    # Check virtual environment
    if (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or hasattr(sys, 'real_prefix'):
        print("✓ Virtual environment is active.")
    else:
        print("⚠ Virtual environment is NOT active. It's highly recommended to use a venv.")

    return True

def create_directory_structure():
    """Create all necessary directories"""
    print("📁 Creating directory structure...")

    directories = [
        "static/css", "static/js", "static/images",
        "templates", "uploads", "logs", "models", "data",
        "charts", "backups", "export", "config"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")

def setup_environment():
    """Setup environment variables and configuration"""
    print("⚙️  Setting up environment...")

    env_file = Path(".env")
    if not env_file.exists():
        print("📝 .env file not found. Creating default .env template...")

        env_template = '''# Industrial Crypto Trading Bot Configuration

# Dashboard Authentication
APP_USER_ID=admin
APP_PASSWORD=CryptoBot2024!

# Alpaca API (for stock trading integration)
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

# Official Alpaca SDK variables (REQUIRED for alpaca-trade-api)
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets

# Cryptocurrency Exchange APIs (Optional - for live trading)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# Bot Configuration
BOT_MODE=paper  # paper or live
INITIAL_BALANCE=10000
MAX_POSITION_SIZE=1000
RISK_PER_TRADE=0.02

# Database Configuration
DATABASE_URL=sqlite:///trading_bot.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# Machine Learning Configuration
ML_TRAINING_INTERVAL=24  # hours
ML_RETRAIN_THRESHOLD=0.05  # accuracy drop threshold

# WebSocket Configuration
WS_HEARTBEAT_INTERVAL=30
WS_RECONNECT_DELAY=5

# Advanced Features
ENABLE_PAPER_TRADING=true
ENABLE_REAL_TRADING=false
ENABLE_ML_PREDICTIONS=true
ENABLE_STRATEGY_OPTIMIZATION=true
ENABLE_RISK_MANAGEMENT=true

# Notification Settings
ENABLE_NOTIFICATIONS=false
DISCORD_WEBHOOK_URL=your_discord_webhook_here
EMAIL_NOTIFICATIONS=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_email_password
'''
        # Remove trailing whitespace and ensure file is written as UTF-8
        env_file.write_text(env_template.strip() + "\n", encoding="utf-8")
        print("  ✓ .env template created. Please update with your credentials before running the bot.")
    else:
        print("  ✓ .env file exists.")

def main():
    try:
        print_banner()
        if not check_system_requirements():
            sys.exit(1)
        create_directory_structure()
        setup_environment()
        print("\n✅ Setup complete! Next steps:")
        print("1. Update your .env file with your actual credentials if you haven't already.")
        print("2. To start the trading bot dashboard/API, run:")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        print("3. Open your browser to http://localhost:8000/")
        print("4. Monitor logs in logs/trading_bot.log")
    except Exception as ex:
        print(f"❌ Unexpected error during setup: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()