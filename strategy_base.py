from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, market_data):
        pass

    @abstractmethod
    def on_order_filled(self, order):
        pass