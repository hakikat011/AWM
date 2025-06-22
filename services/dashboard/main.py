"""
AWM System Dashboard - Main Interface
Real-time monitoring and control interface for the Autonomous Wealth Management system.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared.mcp_client.base import MCPClient

# Page configuration
st.set_page_config(
    page_title="AWM Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        border-left-color: #ff4444 !important;
    }
    .risk-medium {
        border-left-color: #ffaa00 !important;
    }
    .risk-low {
        border-left-color: #00aa44 !important;
    }
    .emergency-mode {
        background-color: #ffebee;
        border: 2px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Server URLs
SERVER_URLS = {
    "portfolio_management": os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_URL", "http://portfolio-management-system:8012"),
    "risk_management": os.getenv("RISK_MANAGEMENT_ENGINE_URL", "http://risk-management-engine:8010"),
    "oms": os.getenv("OMS_URL", "http://order-management-system:8011"),
    "market_data": os.getenv("MARKET_DATA_SERVER_URL", "http://market-data-server:8001"),
}

class DashboardAPI:
    """API client for dashboard data."""
    
    def __init__(self):
        self.client = MCPClient("dashboard")
    
    async def get_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio data."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["portfolio_management"],
                    "get_portfolio",
                    {"portfolio_id": portfolio_id}
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching portfolio data: {e}")
            return {}
    
    async def get_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get portfolio positions."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["portfolio_management"],
                    "get_positions",
                    {"portfolio_id": portfolio_id}
                )
                return response.content.get("positions", [])
        except Exception as e:
            st.error(f"Error fetching positions: {e}")
            return []
    
    async def get_risk_status(self, portfolio_id: str) -> Dict[str, Any]:
        """Get risk status."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "monitor_portfolio_risk",
                    {"portfolio_id": portfolio_id}
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching risk status: {e}")
            return {}
    
    async def get_active_orders(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get active orders."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "get_active_orders",
                    {"portfolio_id": portfolio_id}
                )
                return response.content.get("orders", [])
        except Exception as e:
            st.error(f"Error fetching orders: {e}")
            return []
    
    async def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "get_risk_limits",
                    {}
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching risk limits: {e}")
            return {}

# Initialize API client
@st.cache_resource
def get_api_client():
    return DashboardAPI()

api = get_api_client()

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 AWM Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Autonomous Wealth Management System - Real-time Monitoring & Control**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Portfolio selection
        portfolio_id = st.selectbox(
            "Select Portfolio",
            ["default-portfolio-id"],  # In real implementation, fetch from database
            help="Choose the portfolio to monitor"
        )
        
        # Refresh controls
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.divider()
        
        # System status
        st.subheader("🔧 System Status")
        
        # Check system health
        system_status = check_system_health()
        for service, status in system_status.items():
            if status:
                st.success(f"✅ {service}")
            else:
                st.error(f"❌ {service}")
        
        st.divider()
        
        # Emergency controls
        st.subheader("🚨 Emergency Controls")
        
        # Get risk limits to check emergency mode
        risk_limits = asyncio.run(api.get_risk_limits())
        emergency_mode = risk_limits.get("emergency_mode", False)
        
        if emergency_mode:
            st.markdown(
                '<div class="emergency-mode"><strong>🚨 EMERGENCY MODE ACTIVE</strong><br/>All trading is currently blocked.</div>',
                unsafe_allow_html=True
            )
            if st.button("🔓 Reset Emergency Mode", type="primary"):
                reset_emergency_mode()
        else:
            if st.button("🛑 Emergency Stop", type="secondary"):
                activate_emergency_mode()
    
    # Main content
    if portfolio_id:
        display_portfolio_dashboard(portfolio_id)
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

def check_system_health() -> Dict[str, bool]:
    """Check health of all system components."""
    health_status = {}
    
    for service, url in SERVER_URLS.items():
        try:
            # Simple health check - in real implementation, call health endpoints
            health_status[service] = True  # Simplified for demo
        except:
            health_status[service] = False
    
    return health_status

def display_portfolio_dashboard(portfolio_id: str):
    """Display the main portfolio dashboard."""
    
    # Get data
    portfolio_data = asyncio.run(api.get_portfolio_data(portfolio_id))
    positions = asyncio.run(api.get_positions(portfolio_id))
    risk_status = asyncio.run(api.get_risk_status(portfolio_id))
    active_orders = asyncio.run(api.get_active_orders(portfolio_id))
    
    if not portfolio_data:
        st.error("Unable to load portfolio data")
        return
    
    # Portfolio overview
    st.header("📊 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"₹{portfolio_data.get('current_value', 0):,.2f}",
            f"₹{portfolio_data.get('total_pnl', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            "Available Cash",
            f"₹{portfolio_data.get('available_cash', 0):,.2f}",
            f"{(portfolio_data.get('available_cash', 0) / portfolio_data.get('current_value', 1)) * 100:.1f}%"
        )
    
    with col3:
        total_return = portfolio_data.get('total_return', 0) * 100
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            f"₹{portfolio_data.get('total_pnl', 0):,.2f}"
        )
    
    with col4:
        risk_level = risk_status.get('risk_level', 'UNKNOWN')
        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")
        st.metric(
            "Risk Level",
            f"{risk_color} {risk_level}",
            f"{len(risk_status.get('violations', []))} violations"
        )
    
    # Risk monitoring
    st.header("⚠️ Risk Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk metrics
        risk_metrics = risk_status.get('risk_metrics', {})
        
        st.subheader("Risk Metrics")
        
        if risk_metrics:
            st.metric("VaR (1 Day)", f"{risk_metrics.get('var_1d', 0):.2%}")
            st.metric("Volatility", f"{risk_metrics.get('volatility', 0):.2%}")
            st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
        else:
            st.info("Risk metrics not available")
    
    with col2:
        # Risk violations
        st.subheader("Risk Violations")
        
        violations = risk_status.get('violations', [])
        if violations:
            for violation in violations:
                severity = violation.get('severity', 'MEDIUM')
                st.error(f"**{violation.get('type', 'Unknown')}**: {violation.get('message', 'No details')}")
        else:
            st.success("No risk violations detected")
    
    # Positions
    st.header("💼 Current Positions")
    
    if positions:
        # Create positions DataFrame
        positions_df = pd.DataFrame(positions)
        
        # Display positions table
        st.dataframe(
            positions_df[['symbol', 'quantity', 'average_price', 'current_price', 'market_value', 'unrealized_pnl', 'unrealized_return']],
            use_container_width=True,
            column_config={
                "symbol": "Symbol",
                "quantity": st.column_config.NumberColumn("Quantity", format="%d"),
                "average_price": st.column_config.NumberColumn("Avg Price", format="₹%.2f"),
                "current_price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
                "market_value": st.column_config.NumberColumn("Market Value", format="₹%.2f"),
                "unrealized_pnl": st.column_config.NumberColumn("Unrealized P&L", format="₹%.2f"),
                "unrealized_return": st.column_config.NumberColumn("Return %", format="%.2%"),
            }
        )
        
        # Position allocation chart
        fig_allocation = px.pie(
            positions_df,
            values='market_value',
            names='symbol',
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig_allocation, use_container_width=True)
        
        # P&L chart
        fig_pnl = px.bar(
            positions_df,
            x='symbol',
            y='unrealized_pnl',
            title="Unrealized P&L by Position",
            color='unrealized_pnl',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    else:
        st.info("No positions found")
    
    # Active orders
    st.header("📋 Active Orders")
    
    if active_orders:
        orders_df = pd.DataFrame(active_orders)
        st.dataframe(
            orders_df,
            use_container_width=True,
            column_config={
                "order_id": "Order ID",
                "symbol": "Symbol",
                "side": "Side",
                "quantity": st.column_config.NumberColumn("Quantity", format="%d"),
                "order_type": "Type",
                "status": "Status",
                "created_at": st.column_config.DatetimeColumn("Created At"),
            }
        )
    else:
        st.info("No active orders")
    
    # Trading controls
    st.header("🎯 Trading Controls")
    
    with st.expander("Place New Order"):
        display_order_form(portfolio_id)

def display_order_form(portfolio_id: str):
    """Display order placement form."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.selectbox("Symbol", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"])
        side = st.selectbox("Side", ["BUY", "SELL"])
    
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=1)
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
    
    with col3:
        if order_type == "LIMIT":
            price = st.number_input("Price", min_value=0.01, value=100.0, step=0.01)
        else:
            price = None
        
        st.write("")  # Spacing
        if st.button("Place Order", type="primary"):
            place_order(portfolio_id, symbol, side, quantity, order_type, price)

def place_order(portfolio_id: str, symbol: str, side: str, quantity: int, order_type: str, price: float = None):
    """Place a new order."""
    
    order_request = {
        "portfolio_id": portfolio_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": order_type
    }
    
    if price:
        order_request["price"] = price
    
    try:
        # Place order through OMS
        async def _place_order():
            async with api.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "place_order",
                    order_request
                )
                return response.content
        
        result = asyncio.run(_place_order())
        
        if result.get("status") == "REJECTED":
            st.error(f"Order rejected: {result.get('reason', 'Unknown reason')}")
        else:
            st.success(f"Order placed successfully! Order ID: {result.get('order_id', 'Unknown')}")
            st.rerun()
    
    except Exception as e:
        st.error(f"Error placing order: {e}")

def activate_emergency_mode():
    """Activate emergency mode."""
    try:
        async def _activate():
            async with api.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "emergency_stop",
                    {"reason": "Manual activation from dashboard"}
                )
                return response.content
        
        result = asyncio.run(_activate())
        st.success("Emergency mode activated - All trading stopped!")
        st.rerun()
    
    except Exception as e:
        st.error(f"Error activating emergency mode: {e}")

def reset_emergency_mode():
    """Reset emergency mode."""
    try:
        async def _reset():
            async with api.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "reset_emergency",
                    {"authorized_by": "dashboard_user"}
                )
                return response.content
        
        result = asyncio.run(_reset())
        st.success("Emergency mode reset - Trading resumed!")
        st.rerun()
    
    except Exception as e:
        st.error(f"Error resetting emergency mode: {e}")

if __name__ == "__main__":
    main()
