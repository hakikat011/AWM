"""
Risk Management Dashboard Page
Monitor and control risk management settings and violations.
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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.base import MCPClient

st.set_page_config(
    page_title="Risk Management",
    page_icon="⚠️",
    layout="wide"
)

# Server URLs
SERVER_URLS = {
    "risk_management": os.getenv("RISK_MANAGEMENT_ENGINE_URL", "http://risk-management-engine:8010"),
    "portfolio_management": os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_URL", "http://portfolio-management-system:8012"),
}

class RiskAPI:
    """API client for risk management data."""
    
    def __init__(self):
        self.client = MCPClient("risk_dashboard")
    
    async def get_risk_limits(self) -> dict:
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
    
    async def update_risk_limits(self, updates: dict) -> dict:
        """Update risk limits."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "update_risk_limits",
                    updates
                )
                return response.content
        except Exception as e:
            st.error(f"Error updating risk limits: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def get_risk_violations(self, portfolio_id: str = None) -> dict:
        """Get risk violations."""
        try:
            params = {}
            if portfolio_id:
                params["portfolio_id"] = portfolio_id
            
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "get_risk_violations",
                    params
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching risk violations: {e}")
            return {}
    
    async def monitor_portfolio_risk(self, portfolio_id: str) -> dict:
        """Monitor portfolio risk."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "monitor_portfolio_risk",
                    {"portfolio_id": portfolio_id}
                )
                return response.content
        except Exception as e:
            st.error(f"Error monitoring portfolio risk: {e}")
            return {}
    
    async def emergency_stop(self, reason: str) -> dict:
        """Activate emergency stop."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "emergency_stop",
                    {"reason": reason}
                )
                return response.content
        except Exception as e:
            st.error(f"Error activating emergency stop: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def reset_emergency(self, authorized_by: str) -> dict:
        """Reset emergency mode."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["risk_management"],
                    "reset_emergency",
                    {"authorized_by": authorized_by}
                )
                return response.content
        except Exception as e:
            st.error(f"Error resetting emergency mode: {e}")
            return {"status": "ERROR", "error": str(e)}

@st.cache_resource
def get_risk_api():
    return RiskAPI()

api = get_risk_api()

def main():
    """Main risk management dashboard."""
    
    st.title("⚠️ Risk Management Dashboard")
    st.markdown("**Monitor and control risk management settings and violations**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Risk Controls")
        
        portfolio_id = st.selectbox(
            "Portfolio",
            ["default-portfolio-id"],
            help="Select portfolio for risk monitoring"
        )
        
        if st.button("🔄 Refresh Risk Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Emergency controls
        st.subheader("🚨 Emergency Controls")
        
        # Check emergency mode status
        risk_limits = asyncio.run(api.get_risk_limits())
        emergency_mode = risk_limits.get("emergency_mode", False)
        
        if emergency_mode:
            st.error("🚨 EMERGENCY MODE ACTIVE")
            st.markdown("All trading is currently blocked.")
            
            authorized_by = st.text_input("Authorized By", placeholder="Enter your name")
            if st.button("🔓 Reset Emergency Mode", type="primary"):
                if authorized_by:
                    result = asyncio.run(api.reset_emergency(authorized_by))
                    if result.get("status") != "ERROR":
                        st.success("Emergency mode reset successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to reset: {result.get('error')}")
                else:
                    st.error("Please enter authorization name")
        else:
            reason = st.text_area("Emergency Reason", placeholder="Describe the reason for emergency stop")
            if st.button("🛑 Activate Emergency Stop", type="secondary"):
                if reason:
                    result = asyncio.run(api.emergency_stop(reason))
                    if result.get("status") != "ERROR":
                        st.success("Emergency stop activated!")
                        st.rerun()
                    else:
                        st.error(f"Failed to activate: {result.get('error')}")
                else:
                    st.error("Please provide a reason")
    
    # Main content
    display_risk_dashboard(portfolio_id)

def display_risk_dashboard(portfolio_id: str):
    """Display the main risk dashboard."""
    
    # Get risk data
    risk_limits = asyncio.run(api.get_risk_limits())
    risk_violations = asyncio.run(api.get_risk_violations(portfolio_id))
    portfolio_risk = asyncio.run(api.monitor_portfolio_risk(portfolio_id))
    
    # Risk limits overview
    st.header("📊 Risk Limits Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max Position Size",
            f"₹{risk_limits.get('max_position_size', 0):,.0f}"
        )
    
    with col2:
        st.metric(
            "Max Daily Loss",
            f"₹{risk_limits.get('max_daily_loss', 0):,.0f}"
        )
    
    with col3:
        st.metric(
            "Max Portfolio Risk",
            f"{risk_limits.get('max_portfolio_risk', 0) * 100:.1f}%"
        )
    
    with col4:
        st.metric(
            "Max Leverage",
            f"{risk_limits.get('max_leverage', 0):.1f}x"
        )

    # Current risk status
    st.header("🎯 Current Risk Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Risk Metrics")

        risk_metrics = portfolio_risk.get("risk_metrics", {})
        risk_level = portfolio_risk.get("risk_level", "UNKNOWN")

        # Risk level indicator
        risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}
        risk_color = risk_colors.get(risk_level, "⚪")

        st.markdown(f"**Risk Level:** {risk_color} {risk_level}")

        if risk_metrics:
            st.metric("VaR (1 Day)", f"{abs(risk_metrics.get('var_1d', 0)) * 100:.2f}%")
            st.metric("VaR (5 Days)", f"{abs(risk_metrics.get('var_5d', 0)) * 100:.2f}%")
            st.metric("Expected Shortfall", f"{abs(risk_metrics.get('expected_shortfall', 0)) * 100:.2f}%")
            st.metric("Max Drawdown", f"{abs(risk_metrics.get('max_drawdown', 0)) * 100:.2f}%")
            st.metric("Volatility", f"{risk_metrics.get('volatility', 0) * 100:.2f}%")
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
        else:
            st.info("Risk metrics not available")

    with col2:
        st.subheader("Risk Violations")

        violations = portfolio_risk.get("violations", [])
        all_violations = risk_violations.get("violations", [])

        if violations or all_violations:
            # Current violations
            if violations:
                st.error("**Current Violations:**")
                for violation in violations:
                    st.error(f"• {violation.get('type', 'Unknown')}: {violation.get('current', 'N/A')} > {violation.get('limit', 'N/A')}")

            # Historical violations
            if all_violations:
                st.warning(f"**Total Violations Today:** {len(all_violations)}")

                # Violation types chart
                if isinstance(all_violations, list) and all_violations:
                    violation_types = {}
                    for v in all_violations:
                        v_type = v.get('type', 'Unknown')
                        violation_types[v_type] = violation_types.get(v_type, 0) + 1

                    if violation_types:
                        fig_violations = px.bar(
                            x=list(violation_types.keys()),
                            y=list(violation_types.values()),
                            title="Violation Types",
                            labels={'x': 'Violation Type', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_violations, use_container_width=True)
        else:
            st.success("✅ No risk violations detected")

    # Risk limits configuration
    st.header("⚙️ Risk Limits Configuration")

    with st.expander("Update Risk Limits"):
        display_risk_limits_form(risk_limits)

def display_risk_limits_form(current_limits: dict):
    """Display form to update risk limits."""

    st.subheader("Update Risk Parameters")

    col1, col2 = st.columns(2)

    with col1:
        max_position_size = st.number_input(
            "Max Position Size (₹)",
            min_value=1000,
            value=int(current_limits.get("max_position_size", 100000)),
            step=1000,
            help="Maximum value for a single position"
        )

        max_daily_loss = st.number_input(
            "Max Daily Loss (₹)",
            min_value=1000,
            value=int(current_limits.get("max_daily_loss", 10000)),
            step=1000,
            help="Maximum daily loss allowed"
        )

    with col2:
        max_portfolio_risk = st.number_input(
            "Max Portfolio Risk (%)",
            min_value=0.1,
            max_value=10.0,
            value=current_limits.get("max_portfolio_risk", 0.02) * 100,
            step=0.1,
            help="Maximum portfolio VaR as percentage"
        ) / 100

        max_leverage = st.number_input(
            "Max Leverage",
            min_value=1.0,
            max_value=10.0,
            value=current_limits.get("max_leverage", 3.0),
            step=0.1,
            help="Maximum leverage allowed"
        )

    if st.button("💾 Update Risk Limits", type="primary"):
        updates = {
            "max_position_size": max_position_size,
            "max_daily_loss": max_daily_loss,
            "max_portfolio_risk": max_portfolio_risk,
            "max_leverage": max_leverage
        }

        result = asyncio.run(api.update_risk_limits(updates))

        if result.get("status") == "SUCCESS":
            st.success("Risk limits updated successfully!")
            st.rerun()
        else:
            st.error(f"Failed to update risk limits: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
