"""
Analytics Dashboard Page
Advanced portfolio analytics and performance attribution.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.base import MCPClient

st.set_page_config(
    page_title="AWM Analytics",
    page_icon="📊",
    layout="wide"
)

# Server URLs
SERVER_URLS = {
    "portfolio_management": os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_URL", "http://portfolio-management-system:8012"),
    "market_data": os.getenv("MARKET_DATA_SERVER_URL", "http://market-data-server:8001"),
}

class AnalyticsAPI:
    """API client for analytics data."""
    
    def __init__(self):
        self.client = MCPClient("analytics_dashboard")
    
    async def get_portfolio_analytics(self, portfolio_id: str) -> dict:
        """Get comprehensive portfolio analytics."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["portfolio_management"],
                    "get_analytics",
                    {"portfolio_id": portfolio_id}
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching analytics: {e}")
            return {}
    
    async def get_performance_history(self, portfolio_id: str, days: int) -> dict:
        """Get portfolio performance history."""
        try:
            async with self.client as client:
                response = await client.send_request(
                    SERVER_URLS["portfolio_management"],
                    "get_performance",
                    {"portfolio_id": portfolio_id, "days": days}
                )
                return response.content
        except Exception as e:
            st.error(f"Error fetching performance: {e}")
            return {}

@st.cache_resource
def get_analytics_api():
    return AnalyticsAPI()

api = get_analytics_api()

def main():
    """Main analytics dashboard."""
    
    st.title("📊 Portfolio Analytics")
    st.markdown("**Advanced analytics and performance attribution**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analytics Controls")
        
        portfolio_id = st.selectbox(
            "Portfolio",
            ["default-portfolio-id"],
            help="Select portfolio for analysis"
        )
        
        time_period = st.selectbox(
            "Time Period",
            [30, 60, 90, 180, 365],
            index=2,
            format_func=lambda x: f"{x} days",
            help="Analysis time period"
        )
        
        if st.button("🔄 Refresh Analytics"):
            st.cache_data.clear()
            st.rerun()
    
    if portfolio_id:
        display_analytics_dashboard(portfolio_id, time_period)

def display_analytics_dashboard(portfolio_id: str, days: int):
    """Display comprehensive analytics dashboard."""
    
    # Get analytics data
    analytics = asyncio.run(api.get_portfolio_analytics(portfolio_id))
    performance = asyncio.run(api.get_performance_history(portfolio_id, days))
    
    if not analytics:
        st.error("Unable to load analytics data")
        return
    
    # Portfolio summary
    st.header("📈 Portfolio Summary")
    
    summary = analytics.get("summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"₹{summary.get('total_value', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            "Total P&L",
            f"₹{summary.get('total_pnl', 0):,.2f}",
            f"{summary.get('total_return', 0) * 100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Positions",
            f"{summary.get('number_of_positions', 0)}"
        )
    
    with col4:
        perf_data = analytics.get("performance", {})
        st.metric(
            "Sharpe Ratio",
            f"{perf_data.get('sharpe_ratio', 0):.2f}"
        )
    
    # Performance metrics
    st.header("📊 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk-Adjusted Returns")
        
        perf_metrics = analytics.get("performance", {})
        
        metrics_data = {
            "Metric": ["Total Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
            "Value": [
                f"{perf_metrics.get('total_return', 0) * 100:.2f}%",
                f"{perf_metrics.get('volatility', 0) * 100:.2f}%",
                f"{perf_metrics.get('sharpe_ratio', 0):.2f}",
                f"{perf_metrics.get('max_drawdown', 0) * 100:.2f}%"
            ]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Concentration Analysis")
        
        concentration = analytics.get("concentration", {})
        
        # Concentration metrics
        st.metric("Max Position Weight", f"{concentration.get('max_position_weight', 0) * 100:.1f}%")
        st.metric("Herfindahl Index", f"{concentration.get('herfindahl_index', 0):.3f}")
        st.metric("Top 5 Concentration", f"{concentration.get('top_5_concentration', 0) * 100:.1f}%")
        
        # Risk assessment
        max_weight = concentration.get('max_position_weight', 0)
        if max_weight > 0.3:
            st.error("⚠️ High concentration risk detected")
        elif max_weight > 0.15:
            st.warning("⚠️ Medium concentration risk")
        else:
            st.success("✅ Well diversified portfolio")
    
    # Top performers
    st.header("🏆 Performance Attribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performers")
        
        top_performers = analytics.get("top_performers", [])
        if top_performers:
            top_df = pd.DataFrame(top_performers[:5])
            
            fig_top = px.bar(
                top_df,
                x='symbol',
                y='unrealized_return',
                title="Top 5 Performers",
                color='unrealized_return',
                color_continuous_scale='Greens'
            )
            fig_top.update_layout(showlegend=False)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No performance data available")
    
    with col2:
        st.subheader("Worst Performers")
        
        worst_performers = analytics.get("worst_performers", [])
        if worst_performers:
            worst_df = pd.DataFrame(worst_performers[:5])
            
            fig_worst = px.bar(
                worst_df,
                x='symbol',
                y='unrealized_return',
                title="Bottom 5 Performers",
                color='unrealized_return',
                color_continuous_scale='Reds'
            )
            fig_worst.update_layout(showlegend=False)
            st.plotly_chart(fig_worst, use_container_width=True)
        else:
            st.info("No performance data available")
    
    # Sector allocation
    st.header("🏭 Sector Analysis")
    
    sector_allocation = analytics.get("sector_allocation", {})
    if sector_allocation:
        sector_df = pd.DataFrame(list(sector_allocation.items()), columns=['Sector', 'Weight'])
        
        fig_sector = px.pie(
            sector_df,
            values='Weight',
            names='Sector',
            title="Sector Allocation"
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("Sector allocation data not available")

    # Historical performance simulation
    st.header("📈 Performance Simulation")

    # Generate sample performance data for visualization
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )

    # Simulate portfolio value over time
    np.random.seed(42)  # For consistent demo data
    initial_value = summary.get('total_value', 1000000)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    portfolio_values = [initial_value]

    for ret in returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    performance_df = pd.DataFrame({
        'Date': dates,
        'Portfolio Value': portfolio_values
    })

    # Portfolio value chart
    fig_performance = go.Figure()

    fig_performance.add_trace(go.Scatter(
        x=performance_df['Date'],
        y=performance_df['Portfolio Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))

    fig_performance.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        hovermode='x unified'
    )

    st.plotly_chart(fig_performance, use_container_width=True)

    # Drawdown analysis
    running_max = performance_df['Portfolio Value'].expanding().max()
    drawdown = (performance_df['Portfolio Value'] - running_max) / running_max

    fig_drawdown = go.Figure()

    fig_drawdown.add_trace(go.Scatter(
        x=performance_df['Date'],
        y=drawdown * 100,
        mode='lines',
        name='Drawdown %',
        fill='tonexty',
        line=dict(color='red', width=1)
    ))

    fig_drawdown.update_layout(
        title="Portfolio Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified'
    )

    st.plotly_chart(fig_drawdown, use_container_width=True)

    # Risk metrics over time
    st.header("⚠️ Risk Analysis")

    # Generate sample risk metrics
    risk_dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='D'
    )

    var_1d = np.random.normal(0.02, 0.005, len(risk_dates))
    volatility = np.random.normal(0.15, 0.02, len(risk_dates))

    risk_df = pd.DataFrame({
        'Date': risk_dates,
        'VaR (1D)': var_1d * 100,
        'Volatility': volatility * 100
    })

    fig_risk = go.Figure()

    fig_risk.add_trace(go.Scatter(
        x=risk_df['Date'],
        y=risk_df['VaR (1D)'],
        mode='lines',
        name='VaR (1D) %',
        line=dict(color='orange')
    ))

    fig_risk.add_trace(go.Scatter(
        x=risk_df['Date'],
        y=risk_df['Volatility'],
        mode='lines',
        name='Volatility %',
        line=dict(color='purple'),
        yaxis='y2'
    ))

    fig_risk.update_layout(
        title="Risk Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="VaR (%)",
        yaxis2=dict(
            title="Volatility (%)",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig_risk, use_container_width=True)

if __name__ == "__main__":
    main()
