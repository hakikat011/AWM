"""
Orders Dashboard Page
Monitor and manage orders in the AWM system.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, timezone
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.base import MCPClient

st.set_page_config(
    page_title="Orders Management",
    page_icon="📋",
    layout="wide"
)

# Server URLs
SERVER_URLS = {
    "oms": os.getenv("OMS_URL", "http://order-management-system:8011"),
    "market_data": os.getenv("MARKET_DATA_SERVER_URL", "http://market-data-server:8001"),
}

class OrdersAPI:
    """API client for orders data."""
    
    def __init__(self):
        self.client = MCPClient("orders_dashboard")
    
    async def get_active_orders(self, portfolio_id: str = None) -> dict:
        """Get active orders."""
        try:
            params = {}
            if portfolio_id:
                params["portfolio_id"] = portfolio_id
            
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "get_active_orders",
                    params
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching active orders: {e}")
            return {}
    
    async def get_order_history(self, portfolio_id: str = None, limit: int = 100) -> dict:
        """Get order history."""
        try:
            params = {"limit": limit}
            if portfolio_id:
                params["portfolio_id"] = portfolio_id
            
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "get_order_history",
                    params
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching order history: {e}")
            return {}
    
    async def get_order_status(self, order_id: str) -> dict:
        """Get specific order status."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "get_order_status",
                    {"order_id": order_id}
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching order status: {e}")
            return {}
    
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "cancel_order",
                    {"order_id": order_id}
                )
                return response.content
        except Exception as e:
            st.error(f"Error cancelling order: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def place_order(self, order_request: dict) -> dict:
        """Place a new order."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "place_order",
                    order_request
                )
                return response.content
        except Exception as e:
            st.error(f"Error placing order: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def reconcile_orders(self) -> dict:
        """Reconcile orders with broker."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["oms"],
                    "reconcile_orders",
                    {}
                )
                return response.content
        except Exception as e:
            st.error(f"Error reconciling orders: {e}")
            return {"status": "ERROR", "error": str(e)}

@st.cache_resource
def get_orders_api():
    return OrdersAPI()

api = get_orders_api()

def main():
    """Main orders dashboard."""
    
    st.title("📋 Orders Management")
    st.markdown("**Monitor and manage orders in the AWM system**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Order Controls")
        
        portfolio_id = st.selectbox(
            "Portfolio",
            ["All Portfolios", "default-portfolio-id"],
            help="Select portfolio to filter orders"
        )
        
        if portfolio_id == "All Portfolios":
            portfolio_id = None
        
        view_mode = st.selectbox(
            "View Mode",
            ["Active Orders", "Order History", "Order Analytics"],
            help="Choose what to display"
        )
        
        if st.button("🔄 Refresh Orders"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Order actions
        st.subheader("Order Actions")
        
        if st.button("🔄 Reconcile Orders"):
            result = asyncio.run(api.reconcile_orders())
            if result.get("status") == "SUCCESS":
                st.success("Orders reconciled successfully!")
            else:
                st.error(f"Reconciliation failed: {result.get('error')}")
        
        st.divider()
        
        # Quick order placement
        with st.expander("Quick Order"):
            display_quick_order_form(portfolio_id or "default-portfolio-id")
    
    # Main content based on view mode
    if view_mode == "Active Orders":
        display_active_orders(portfolio_id)
    elif view_mode == "Order History":
        display_order_history(portfolio_id)
    else:
        display_order_analytics(portfolio_id)

def display_active_orders(portfolio_id: str = None):
    """Display active orders."""
    
    st.header("🔄 Active Orders")
    
    # Get active orders
    orders_data = asyncio.run(api.get_active_orders(portfolio_id))
    orders = orders_data.get("orders", [])
    
    if orders:
        # Orders summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Active Orders", len(orders))
        
        with col2:
            buy_orders = len([o for o in orders if o.get("side") == "BUY"])
            st.metric("Buy Orders", buy_orders)
        
        with col3:
            sell_orders = len([o for o in orders if o.get("side") == "SELL"])
            st.metric("Sell Orders", sell_orders)
        
        with col4:
            pending_orders = len([o for o in orders if o.get("status") == "PENDING"])
            st.metric("Pending Orders", pending_orders)
        
        # Orders table
        st.subheader("Active Orders Details")
        
        orders_df = pd.DataFrame(orders)
        
        # Add action buttons
        for i, order in enumerate(orders):
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                # Order details
                st.write(f"**{order.get('symbol')}** - {order.get('side')} {order.get('quantity')} @ {order.get('order_type')}")
                st.write(f"Status: {order.get('status')} | Created: {order.get('created_at')}")
            
            with col2:
                if st.button("📊", key=f"details_{i}", help="View Details"):
                    display_order_details(order.get('order_id'))
            
            with col3:
                if order.get('status') in ['PENDING', 'OPEN']:
                    if st.button("❌", key=f"cancel_{i}", help="Cancel Order"):
                        cancel_order_action(order.get('order_id'))
        
        # Orders by status chart
        if len(orders) > 0:
            status_counts = orders_df['status'].value_counts()
            
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Orders by Status"
            )
            st.plotly_chart(fig_status, use_container_width=True)
    
    else:
        st.info("No active orders found")

def display_quick_order_form(portfolio_id: str):
    """Display quick order placement form."""
    
    st.subheader("Quick Order")
    
    symbol = st.selectbox("Symbol", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        side = st.selectbox("Side", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", min_value=1, value=1)
    
    with col2:
        order_type = st.selectbox("Type", ["MARKET", "LIMIT"])
        if order_type == "LIMIT":
            price = st.number_input("Price", min_value=0.01, value=100.0, step=0.01)
        else:
            price = None
    
    if st.button("🚀 Place Order", type="primary"):
        place_quick_order(portfolio_id, symbol, side, quantity, order_type, price)

def place_quick_order(portfolio_id: str, symbol: str, side: str, quantity: int, order_type: str, price: float = None):
    """Place a quick order."""
    
    order_request = {
        "portfolio_id": portfolio_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": order_type
    }
    
    if price:
        order_request["price"] = price
    
    result = asyncio.run(api.place_order(order_request))
    
    if result.get("status") == "REJECTED":
        st.error(f"Order rejected: {result.get('reason', 'Unknown reason')}")
    elif result.get("status") == "ERROR":
        st.error(f"Order failed: {result.get('error', 'Unknown error')}")
    else:
        st.success(f"Order placed! ID: {result.get('order_id', 'Unknown')}")
        st.rerun()

if __name__ == "__main__":
    main()
