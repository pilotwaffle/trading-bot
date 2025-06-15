"""
WebSocket Manager for Real-time Dashboard Updates
Handles all WebSocket connections for the trading bot
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class WSMessage:
    type: str
    data: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates
    """
    
    def __init__(self):
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Connection metadata
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Subscription management
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        
        # Message queue for offline connections
        self.message_queue: Dict[str, List[WSMessage]] = {}
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'messages_sent': 0,
            'messages_failed': 0,
            'uptime': datetime.now()
        }
        
        # Heartbeat
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_task = None
        
        logger.info("WebSocket ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            self.active_connections.append(websocket)
            
            # Store connection info
            self.connection_info[websocket] = {
                'client_id': client_id or f"client_{len(self.active_connections)}",
                'connected_at': datetime.now(),
                'last_ping': datetime.now(),
                'subscriptions': set()
            }
            
            self.stats['total_connections'] += 1
            
            # Send welcome message
            await self.send_personal_message(websocket, WSMessage(
                type="connection_established",
                data={
                    "client_id": self.connection_info[websocket]['client_id'],
                    "server_time": datetime.now().isoformat(),
                    "available_channels": list(self.subscriptions.keys())
                }
            ))
            
            # Start heartbeat if not already running
            if self.heartbeat_task is None:
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"WebSocket connected: {self.connection_info[websocket]['client_id']}")
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            # Clean up subscriptions
            if websocket in self.connection_info:
                client_id = self.connection_info[websocket]['client_id']
                
                # Remove from all subscription channels
                for channel_subs in self.subscriptions.values():
                    channel_subs.discard(websocket)
                
                # Clean up connection info
                del self.connection_info[websocket]
                
                logger.info(f"WebSocket disconnected: {client_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_personal_message(self, websocket: WebSocket, message: WSMessage):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message.to_dict()))
            self.stats['messages_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.stats['messages_failed'] += 1
            
            # Remove broken connection
            self.disconnect(websocket)
    
    async def broadcast(self, message: WSMessage, channel: str = None):
        """Broadcast message to all or specific channel subscribers"""
        try:
            # Determine target connections
            if channel and channel in self.subscriptions:
                target_connections = list(self.subscriptions[channel])
            else:
                target_connections = self.active_connections.copy()
            
            if not target_connections:
                return
            
            # Send to all target connections
            disconnected_connections = []
            message_json = json.dumps(message.to_dict())
            
            for connection in target_connections:
                try:
                    await connection.send_text(message_json)
                    self.stats['messages_sent'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to send message to connection: {e}")
                    disconnected_connections.append(connection)
                    self.stats['messages_failed'] += 1
            
            # Clean up disconnected connections
            for connection in disconnected_connections:
                self.disconnect(connection)
            
            logger.debug(f"Broadcast message to {len(target_connections)} connections")
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe WebSocket to a specific channel"""
        try:
            if channel not in self.subscriptions:
                self.subscriptions[channel] = set()
            
            self.subscriptions[channel].add(websocket)
            
            if websocket in self.connection_info:
                self.connection_info[websocket]['subscriptions'].add(channel)
            
            await self.send_personal_message(websocket, WSMessage(
                type="subscription_confirmed",
                data={"channel": channel}
            ))
            
            logger.info(f"WebSocket subscribed to {channel}")
            
        except Exception as e:
            logger.error(f"Error subscribing to channel: {e}")
    
    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe WebSocket from a specific channel"""
        try:
            if channel in self.subscriptions:
                self.subscriptions[channel].discard(websocket)
            
            if websocket in self.connection_info:
                self.connection_info[websocket]['subscriptions'].discard(channel)
            
            await self.send_personal_message(websocket, WSMessage(
                type="unsubscription_confirmed",
                data={"channel": channel}
            ))
            
            logger.info(f"WebSocket unsubscribed from {channel}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from channel: {e}")
    
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            msg_data = data.get('data', {})
            
            if msg_type == "subscribe":
                channel = msg_data.get('channel')
                if channel:
                    await self.subscribe(websocket, channel)
                    
            elif msg_type == "unsubscribe":
                channel = msg_data.get('channel')
                if channel:
                    await self.unsubscribe(websocket, channel)
                    
            elif msg_type == "ping":
                await self.send_personal_message(websocket, WSMessage(
                    type="pong",
                    data={"server_time": datetime.now().isoformat()}
                ))
                
                # Update last ping time
                if websocket in self.connection_info:
                    self.connection_info[websocket]['last_ping'] = datetime.now()
                    
            elif msg_type == "get_status":
                await self.send_personal_message(websocket, WSMessage(
                    type="status_response",
                    data=self.get_status()
                ))
                
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON in WebSocket message")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat to check connection health"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.active_connections:
                    # No active connections, stop heartbeat
                    self.heartbeat_task = None
                    break
                
                # Send heartbeat to all connections
                heartbeat_msg = WSMessage(
                    type="heartbeat",
                    data={"server_time": datetime.now().isoformat()}
                )
                
                disconnected = []
                for connection in self.active_connections.copy():
                    try:
                        await connection.send_text(json.dumps(heartbeat_msg.to_dict()))
                    except:
                        disconnected.append(connection)
                
                # Clean up disconnected connections
                for connection in disconnected:
                    self.disconnect(connection)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection manager status"""
        uptime = datetime.now() - self.stats['uptime']
        
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.stats['total_connections'],
            "messages_sent": self.stats['messages_sent'],
            "messages_failed": self.stats['messages_failed'],
            "uptime_seconds": uptime.total_seconds(),
            "channels": {
                channel: len(subscribers) 
                for channel, subscribers in self.subscriptions.items()
            },
            "server_time": datetime.now().isoformat()
        }
    
    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections"""
        info = []
        for websocket, data in self.connection_info.items():
            info.append({
                "client_id": data['client_id'],
                "connected_at": data['connected_at'].isoformat(),
                "last_ping": data['last_ping'].isoformat(),
                "subscriptions": list(data['subscriptions']),
                "is_active": websocket in self.active_connections
            })
        return info

class TradingBotWebSocketManager:
    """
    Specialized WebSocket manager for trading bot events
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        
        # Define trading bot specific channels
        self.channels = {
            'market_data': 'Real-time market price updates',
            'trades': 'Trade execution notifications',
            'positions': 'Position updates',
            'alerts': 'Trading alerts and notifications',
            'strategy': 'Strategy status updates',
            'portfolio': 'Portfolio performance updates',
            'system': 'System status and health updates'
        }
        
        logger.info("Trading Bot WebSocket Manager initialized")
    
    async def broadcast_market_update(self, market_data: Dict[str, Any]):
        """Broadcast market data update"""
        message = WSMessage(
            type="market_update",
            data=market_data
        )
        await self.connection_manager.broadcast(message, "market_data")
    
    async def broadcast_trade_executed(self, trade_data: Dict[str, Any]):
        """Broadcast trade execution"""
        message = WSMessage(
            type="trade_executed",
            data=trade_data
        )
        await self.connection_manager.broadcast(message, "trades")
    
    async def broadcast_position_update(self, position_data: Dict[str, Any]):
        """Broadcast position update"""
        message = WSMessage(
            type="position_update",
            data=position_data
        )
        await self.connection_manager.broadcast(message, "positions")
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast trading alert"""
        message = WSMessage(
            type="alert",
            data=alert_data
        )
        await self.connection_manager.broadcast(message, "alerts")
    
    async def broadcast_strategy_update(self, strategy_data: Dict[str, Any]):
        """Broadcast strategy status update"""
        message = WSMessage(
            type="strategy_update",
            data=strategy_data
        )
        await self.connection_manager.broadcast(message, "strategy")
    
    async def broadcast_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Broadcast portfolio performance update"""
        message = WSMessage(
            type="portfolio_update",
            data=portfolio_data
        )
        await self.connection_manager.broadcast(message, "portfolio")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        message = WSMessage(
            type="system_status",
            data=status_data
        )
        await self.connection_manager.broadcast(message, "system")
    
    async def send_notification(self, client_id: str, notification: Dict[str, Any]):
        """Send notification to specific client"""
        # Find client connection
        target_connection = None
        for ws, info in self.connection_manager.connection_info.items():
            if info['client_id'] == client_id:
                target_connection = ws
                break
        
        if target_connection:
            message = WSMessage(
                type="notification",
                data=notification
            )
            await self.connection_manager.send_personal_message(target_connection, message)
    
    def get_channel_info(self) -> Dict[str, str]:
        """Get information about available channels"""
        return self.channels.copy()

# Global instances
_connection_manager = None
_trading_ws_manager = None

def get_connection_manager() -> ConnectionManager:
    """Get singleton connection manager"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager

def get_trading_websocket_manager() -> TradingBotWebSocketManager:
    """Get singleton trading WebSocket manager"""
    global _trading_ws_manager
    if _trading_ws_manager is None:
        cm = get_connection_manager()
        _trading_ws_manager = TradingBotWebSocketManager(cm)
    return _trading_ws_manager

# FastAPI WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """Main WebSocket endpoint for FastAPI"""
    manager = get_connection_manager()
    
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Wait for message from client
            data = await websocket.receive_text()
            await manager.handle_message(websocket, data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Test the WebSocket manager