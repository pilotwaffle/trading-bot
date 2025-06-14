#!/usr/bin/env python3
"""
Fixed Quick Start Script for Crypto Trading Bot
Comprehensive dependency checking and initialization
"""

import os
import sys
import logging
import asyncio
import importlib
import traceback
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class ImportResult:
    success: bool
    module: str
    version: Optional[str] = None
    error: Optional[str] = None

class ComprehensiveImportChecker:
    def __init__(self):
        self.logger = self._setup_logging()
        self.results: Dict[str, ImportResult] = {}

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/import_check.log', mode='w')
            ]
        )
        return logging.getLogger(__name__)

    def check_all_dependencies(self) -> bool:
        critical_modules = {
            'ccxt': self._check_ccxt,
            'fastapi': self._check_fastapi,
            'uvicorn': self._check_uvicorn,
            'pandas': self._check_pandas,
            'numpy': self._check_numpy,
            'aiohttp': self._check_aiohttp,
            'websockets': self._check_websockets,
            'cryptography': self._check_cryptography,
            'pydantic': self._check_pydantic,
            'requests': self._check_requests
        }
        optional_modules = {
            'yfinance': self._check_standard_module,
            'beautifulsoup4': self._check_bs4,
            'python-dateutil': self._check_dateutil,
            'pytz': self._check_standard_module,
            'loguru': self._check_standard_module
        }
        self.logger.info("🔍 Starting comprehensive dependency check...")
        if not self._check_python_version():
            return False
        critical_passed = True
        for module_name, check_func in critical_modules.items():
            try:
                result = check_func(module_name) if check_func == self._check_standard_module else check_func()
                self.results[module_name] = result
                if result.success:
                    self.logger.info(f"✅ {module_name} v{result.version} - OK")
                else:
                    self.logger.error(f"❌ {module_name} - FAILED: {result.error}")
                    critical_passed = False
            except Exception as e:
                error_msg = f"Unexpected error checking {module_name}: {str(e)}"
                self.logger.error(f"❌ {module_name} - ERROR: {error_msg}")
                self.results[module_name] = ImportResult(False, module_name, error=error_msg)
                critical_passed = False
        for module_name, check_func in optional_modules.items():
            try:
                result = check_func(module_name) if check_func == self._check_standard_module else check_func()
                self.results[module_name] = result
                if result.success:
                    self.logger.info(f"✅ {module_name} v{result.version} - OK (optional)")
                else:
                    self.logger.warning(f"⚠️  {module_name} - MISSING (optional): {result.error}")
            except Exception as e:
                self.logger.warning(f"⚠️  {module_name} - WARNING: {str(e)}")
        return critical_passed

    def _check_python_version(self, min_version=(3, 8)) -> bool:
        current_version = sys.version_info[:2]
        if current_version < min_version:
            self.logger.error(
                f"❌ Python {min_version[0]}.{min_version[1]}+ required, "
                f"but {current_version[0]}.{current_version[1]} found"
            )
            return False
        self.logger.info(f"✅ Python {current_version[0]}.{current_version[1]} - OK")
        return True

    def _check_ccxt(self) -> ImportResult:
        try:
            import ccxt
            exchange = ccxt.binance()
            markets = exchange.load_markets()
            if not markets:
                raise Exception("No markets loaded")
            return ImportResult(True, "ccxt", version=ccxt.__version__)
        except ImportError as e:
            return ImportResult(False, "ccxt", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "ccxt", error=f"CCXT test failed: {str(e)}")

    def _check_fastapi(self) -> ImportResult:
        try:
            import fastapi
            app = fastapi.FastAPI()
            return ImportResult(True, "fastapi", version=fastapi.__version__)
        except ImportError as e:
            return ImportResult(False, "fastapi", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "fastapi", error=f"FastAPI test failed: {str(e)}")

    def _check_uvicorn(self) -> ImportResult:
        try:
            import uvicorn
            return ImportResult(True, "uvicorn", version=uvicorn.__version__)
        except ImportError as e:
            return ImportResult(False, "uvicorn", error=f"Import failed: {str(e)}")

    def _check_pandas(self) -> ImportResult:
        try:
            import pandas as pd
            df = pd.DataFrame({'test': [1, 2, 3]})
            assert len(df) == 3
            return ImportResult(True, "pandas", version=pd.__version__)
        except ImportError as e:
            return ImportResult(False, "pandas", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "pandas", error=f"Pandas test failed: {str(e)}")

    def _check_numpy(self) -> ImportResult:
        try:
            import numpy as np
            arr = np.array([1, 2, 3])
            assert arr.sum() == 6
            return ImportResult(True, "numpy", version=np.__version__)
        except ImportError as e:
            return ImportResult(False, "numpy", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "numpy", error=f"Numpy test failed: {str(e)}")

    def _check_aiohttp(self) -> ImportResult:
        try:
            import aiohttp
            return ImportResult(True, "aiohttp", version=aiohttp.__version__)
        except ImportError as e:
            return ImportResult(False, "aiohttp", error=f"Import failed: {str(e)}")

    def _check_websockets(self) -> ImportResult:
        try:
            import websockets
            version_parts = websockets.__version__.split('.')
            major_version = int(version_parts[0])
            if major_version >= 15:
                self.logger.warning(f"⚠️  WebSockets v{websockets.__version__} may conflict with alpaca-trade-api")
            return ImportResult(True, "websockets", version=websockets.__version__)
        except ImportError as e:
            return ImportResult(False, "websockets", error=f"Import failed: {str(e)}")

    def _check_cryptography(self) -> ImportResult:
        try:
            import cryptography
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            f = Fernet(key)
            return ImportResult(True, "cryptography", version=cryptography.__version__)
        except ImportError as e:
            return ImportResult(False, "cryptography", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "cryptography", error=f"Cryptography test failed: {str(e)}")

    def _check_pydantic(self) -> ImportResult:
        try:
            import pydantic
            from pydantic import BaseModel
            class TestModel(BaseModel):
                name: str
                age: int
            test = TestModel(name="test", age=25)
            return ImportResult(True, "pydantic", version=pydantic.__version__)
        except ImportError as e:
            return ImportResult(False, "pydantic", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "pydantic", error=f"Pydantic test failed: {str(e)}")

    def _check_requests(self) -> ImportResult:
        try:
            import requests
            return ImportResult(True, "requests", version=requests.__version__)
        except ImportError as e:
            return ImportResult(False, "requests", error=f"Import failed: {str(e)}")

    def _check_bs4(self, module_name="beautifulsoup4") -> ImportResult:
        try:
            import bs4
            from bs4 import BeautifulSoup
            soup = BeautifulSoup("<html><body>test</body></html>", 'html.parser')
            return ImportResult(True, "beautifulsoup4", version=bs4.__version__)
        except ImportError as e:
            return ImportResult(False, "beautifulsoup4", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "beautifulsoup4", error=f"BeautifulSoup test failed: {str(e)}")

    def _check_dateutil(self, module_name="python-dateutil") -> ImportResult:
        try:
            import dateutil
            from dateutil import parser
            date = parser.parse("2023-01-01")
            return ImportResult(True, "python-dateutil", version=dateutil.__version__)
        except ImportError as e:
            return ImportResult(False, "python-dateutil", error=f"Import failed: {str(e)}")
        except Exception as e:
            return ImportResult(False, "python-dateutil", error=f"Dateutil test failed: {str(e)}")

    def _check_standard_module(self, module_name: str) -> ImportResult:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            return ImportResult(True, module_name, version=version)
        except ImportError as e:
            return ImportResult(False, module_name, error=f"Import failed: {str(e)}")

    def generate_fix_instructions(self) -> None:
        failed_modules = [name for name, result in self.results.items() if not result.success]
        if not failed_modules:
            self.logger.info("🎉 All dependencies are properly installed!")
            return
        self.logger.error("\n" + "="*60)
        self.logger.error("❌ MISSING DEPENDENCIES DETECTED")
        self.logger.error("="*60)
        pip_installs = []
        special_instructions = {}
        for module in failed_modules:
            if module == "TA-Lib":
                special_instructions[module] = [
                    "# TA-Lib requires system dependencies:",
                    "# On Ubuntu/Debian: sudo apt-get install libta-lib-dev",
                    "# On MacOS: brew install ta-lib",
                    "# On Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib",
                    "pip install TA-Lib"
                ]
            elif module == "beautifulsoup4":
                pip_installs.append("beautifulsoup4 lxml")
            elif module == "python-dateutil":
                pip_installs.append("python-dateutil")
            else:
                pip_installs.append(module)
        if pip_installs:
            self.logger.error("\n🔧 Run these commands to fix missing dependencies:")
            self.logger.error(f"pip install {' '.join(pip_installs)}")
        for module, instructions in special_instructions.items():
            self.logger.error(f"\n🔧 Special installation for {module}:")
            for instruction in instructions:
                self.logger.error(instruction)
        self.logger.error("\n" + "="*60)

class TradingBotInitializer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        try:
            os.makedirs("logs", exist_ok=True)
            os.makedirs("data", exist_ok=True)
            os.makedirs("config", exist_ok=True)
            checker = ComprehensiveImportChecker()
            if not checker.check_all_dependencies():
                checker.generate_fix_instructions()
                return False
            self.logger.info("✅ All dependency checks passed!")
            self.logger.info("🚀 Trading bot initialized successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {str(e)}")
            traceback.print_exc()
            return False

async def main():
    print("🚀 Crypto Trading Bot - Quick Start")
    print("="*50)
    initializer = TradingBotInitializer()
    try:
        success = await initializer.initialize()
        if success:
            print("\n🎉 SUCCESS! Your crypto trading bot is ready to use!")
            print("\nNext steps:")
            print("1. Configure your API keys in config/settings.py or .env")
            print("2. Set up your trading strategies")
            print("3. Run your bot with: python main.py")
            print("\n📊 Visit http://localhost:8000/docs for API documentation")
        else:
            print("\n❌ SETUP FAILED! Please fix the issues above and try again.")
            print("\nNeed help? Check the logs in the 'logs' directory.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())