"""
AI Agents Dashboard Page
Monitor and control AI agents in the AWM system.
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
from shared.database.connection import db_manager, init_database

st.set_page_config(
    page_title="AI Agents",
    page_icon="🤖",
    layout="wide"
)

# Agent URLs (these would be actual agent endpoints in production)
AGENT_URLS = {
    "market_analysis": "http://market-analysis-agent:8020",
    "risk_assessment": "http://risk-assessment-agent:8021",
    "strategy_optimization": "http://strategy-optimization-agent:8022",
    "trade_execution": "http://trade-execution-agent:8023"
}

class AgentAPI:
    """API client for agent monitoring."""
    
    def __init__(self):
        self.client = MCPClient("agent_dashboard")
    
    async def get_agent_status(self, agent_name: str) -> dict:
        """Get agent health status."""
        try:
            # In a real implementation, this would call the agent's health endpoint
            # For now, we'll simulate agent status
            return {
                "agent_name": agent_name,
                "status": "RUNNING",
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "tasks_completed": 150,
                "tasks_failed": 2,
                "average_processing_time": 2.5,
                "queue_size": 3
            }
        except Exception as e:
            return {
                "agent_name": agent_name,
                "status": "ERROR",
                "error": str(e)
            }
    
    async def get_agent_tasks(self, agent_name: str = None) -> list:
        """Get agent task history."""
        try:
            # In a real implementation, this would query the agent_tasks table
            # For now, we'll return sample data
            sample_tasks = [
                {
                    "id": "task-1",
                    "agent_name": "market_analysis_agent",
                    "task_type": "analyze_instrument",
                    "status": "COMPLETED",
                    "created_at": datetime.now() - timedelta(minutes=30),
                    "completed_at": datetime.now() - timedelta(minutes=28),
                    "processing_time": 120
                },
                {
                    "id": "task-2",
                    "agent_name": "risk_assessment_agent",
                    "task_type": "assess_portfolio_risk",
                    "status": "COMPLETED",
                    "created_at": datetime.now() - timedelta(minutes=25),
                    "completed_at": datetime.now() - timedelta(minutes=23),
                    "processing_time": 90
                },
                {
                    "id": "task-3",
                    "agent_name": "strategy_optimization_agent",
                    "task_type": "backtest_strategy",
                    "status": "IN_PROGRESS",
                    "created_at": datetime.now() - timedelta(minutes=10),
                    "completed_at": None,
                    "processing_time": None
                }
            ]
            
            if agent_name:
                return [task for task in sample_tasks if task["agent_name"] == agent_name]
            return sample_tasks
            
        except Exception as e:
            st.error(f"Error fetching agent tasks: {e}")
            return []
    
    async def trigger_agent_task(self, agent_name: str, task_type: str, parameters: dict) -> dict:
        """Trigger a new agent task."""
        try:
            # In a real implementation, this would send a task to the agent
            return {
                "status": "SUCCESS",
                "task_id": f"task-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "message": f"Task {task_type} submitted to {agent_name}"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }

@st.cache_resource
def get_agent_api():
    return AgentAPI()

api = get_agent_api()

def main():
    """Main AI agents dashboard."""
    
    st.title("🤖 AI Agents Dashboard")
    st.markdown("**Monitor and control AI agents in the AWM system**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Agent Controls")
        
        selected_agent = st.selectbox(
            "Select Agent",
            ["All Agents"] + list(AGENT_URLS.keys()),
            help="Choose agent to monitor"
        )
        
        if st.button("🔄 Refresh Status"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Agent actions
        st.subheader("Agent Actions")
        
        if selected_agent != "All Agents":
            if st.button("▶️ Start Agent"):
                st.success(f"Start command sent to {selected_agent}")
            
            if st.button("⏸️ Pause Agent"):
                st.warning(f"Pause command sent to {selected_agent}")
            
            if st.button("🔄 Restart Agent"):
                st.info(f"Restart command sent to {selected_agent}")
    
    # Main content
    if selected_agent == "All Agents":
        display_all_agents_overview()
    else:
        display_agent_details(selected_agent)

def display_all_agents_overview():
    """Display overview of all agents."""
    
    st.header("🔍 Agents Overview")
    
    # Get status for all agents
    agent_statuses = []
    for agent_name in AGENT_URLS.keys():
        status = asyncio.run(api.get_agent_status(agent_name))
        agent_statuses.append(status)
    
    # Agent status cards
    cols = st.columns(2)
    
    for i, status in enumerate(agent_statuses):
        with cols[i % 2]:
            agent_name = status["agent_name"]
            agent_status = status["status"]
            
            # Status color
            if agent_status == "RUNNING":
                status_color = "🟢"
            elif agent_status == "ERROR":
                status_color = "🔴"
            else:
                status_color = "🟡"
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h4>{status_color} {agent_name.replace('_', ' ').title()}</h4>
                <p><strong>Status:</strong> {agent_status}</p>
                <p><strong>Tasks Completed:</strong> {status.get('tasks_completed', 0)}</p>
                <p><strong>Queue Size:</strong> {status.get('queue_size', 0)}</p>
                <p><strong>Avg Processing:</strong> {status.get('average_processing_time', 0):.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Agent performance metrics
    st.header("📊 Agent Performance")
    
    # Create performance DataFrame
    perf_data = []
    for status in agent_statuses:
        perf_data.append({
            "Agent": status["agent_name"].replace('_', ' ').title(),
            "Tasks Completed": status.get("tasks_completed", 0),
            "Tasks Failed": status.get("tasks_failed", 0),
            "Success Rate": (status.get("tasks_completed", 0) / max(status.get("tasks_completed", 0) + status.get("tasks_failed", 0), 1)) * 100,
            "Avg Processing Time": status.get("average_processing_time", 0)
        })
    
    perf_df = pd.DataFrame(perf_data)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tasks = px.bar(
            perf_df,
            x="Agent",
            y="Tasks Completed",
            title="Tasks Completed by Agent",
            color="Tasks Completed"
        )
        st.plotly_chart(fig_tasks, use_container_width=True)
    
    with col2:
        fig_success = px.bar(
            perf_df,
            x="Agent",
            y="Success Rate",
            title="Success Rate by Agent (%)",
            color="Success Rate",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_success, use_container_width=True)

    # Recent tasks
    st.header("📋 Recent Agent Tasks")

    recent_tasks = asyncio.run(api.get_agent_tasks())

    if recent_tasks:
        tasks_df = pd.DataFrame(recent_tasks)

        # Format datetime columns
        tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
        tasks_df['completed_at'] = pd.to_datetime(tasks_df['completed_at'])

        st.dataframe(
            tasks_df[['agent_name', 'task_type', 'status', 'created_at', 'processing_time']],
            use_container_width=True,
            column_config={
                "agent_name": "Agent",
                "task_type": "Task Type",
                "status": "Status",
                "created_at": st.column_config.DatetimeColumn("Created At"),
                "processing_time": st.column_config.NumberColumn("Processing Time (s)", format="%.1f")
            }
        )
    else:
        st.info("No recent tasks found")

def display_agent_details(agent_name: str):
    """Display detailed view of a specific agent."""

    st.header(f"🤖 {agent_name.replace('_', ' ').title()} Agent")

    # Get agent status
    status = asyncio.run(api.get_agent_status(agent_name))

    # Agent status overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_color = "🟢" if status["status"] == "RUNNING" else "🔴"
        st.metric("Status", f"{status_color} {status['status']}")

    with col2:
        st.metric("Tasks Completed", status.get("tasks_completed", 0))

    with col3:
        st.metric("Queue Size", status.get("queue_size", 0))

    with col4:
        st.metric("Avg Processing", f"{status.get('average_processing_time', 0):.1f}s")

    # Agent-specific controls
    st.header("🎛️ Agent Controls")

    with st.expander("Trigger New Task"):
        display_task_form(agent_name)

    # Agent tasks history
    st.header("📋 Task History")

    agent_tasks = asyncio.run(api.get_agent_tasks(agent_name))

    if agent_tasks:
        tasks_df = pd.DataFrame(agent_tasks)

        # Task status distribution
        status_counts = tasks_df['status'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Task Status Distribution"
            )
            st.plotly_chart(fig_status, use_container_width=True)

        with col2:
            # Processing time distribution
            completed_tasks = tasks_df[tasks_df['status'] == 'COMPLETED']
            if not completed_tasks.empty:
                fig_time = px.histogram(
                    completed_tasks,
                    x='processing_time',
                    title="Processing Time Distribution",
                    nbins=10
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("No completed tasks for processing time analysis")

        # Detailed task list
        st.subheader("Task Details")

        tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])

        st.dataframe(
            tasks_df,
            use_container_width=True,
            column_config={
                "id": "Task ID",
                "task_type": "Task Type",
                "status": "Status",
                "created_at": st.column_config.DatetimeColumn("Created At"),
                "completed_at": st.column_config.DatetimeColumn("Completed At"),
                "processing_time": st.column_config.NumberColumn("Processing Time (s)", format="%.1f")
            }
        )
    else:
        st.info(f"No tasks found for {agent_name}")

def display_task_form(agent_name: str):
    """Display form to trigger new agent tasks."""

    # Task type options based on agent
    task_options = {
        "market_analysis": ["analyze_instrument", "scan_market", "generate_signals"],
        "risk_assessment": ["assess_portfolio_risk", "calculate_position_size", "evaluate_trade_risk"],
        "strategy_optimization": ["backtest_strategy", "optimize_parameters", "compare_strategies"],
        "trade_execution": ["execute_trade", "optimize_execution", "monitor_orders"]
    }

    task_type = st.selectbox(
        "Task Type",
        task_options.get(agent_name, ["custom_task"]),
        help="Select the type of task to execute"
    )

    # Parameters based on task type
    parameters = {}

    if task_type == "analyze_instrument":
        symbol = st.text_input("Symbol", value="RELIANCE", help="Stock symbol to analyze")
        analysis_type = st.selectbox("Analysis Type", ["comprehensive", "technical", "fundamental"])
        parameters = {"symbol": symbol, "analysis_type": analysis_type}

    elif task_type == "assess_portfolio_risk":
        portfolio_id = st.text_input("Portfolio ID", value="default-portfolio-id")
        parameters = {"portfolio_id": portfolio_id}

    elif task_type == "backtest_strategy":
        strategy = st.selectbox("Strategy", ["sma_crossover", "rsi_mean_reversion", "bollinger_bands"])
        symbol = st.text_input("Symbol", value="RELIANCE")
        parameters = {
            "strategy": strategy,
            "symbol": symbol,
            "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "end_date": datetime.now().isoformat()
        }

    elif task_type == "execute_trade":
        portfolio_id = st.text_input("Portfolio ID", value="default-portfolio-id")
        symbol = st.text_input("Symbol", value="RELIANCE")
        side = st.selectbox("Side", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", min_value=1, value=10)
        parameters = {
            "trade_proposal": {
                "portfolio_id": portfolio_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": 2500
            }
        }

    else:
        # Custom parameters
        st.text_area("Parameters (JSON)", value="{}", help="Enter task parameters as JSON")

    if st.button("🚀 Submit Task", type="primary"):
        result = asyncio.run(api.trigger_agent_task(agent_name, task_type, parameters))

        if result["status"] == "SUCCESS":
            st.success(f"Task submitted successfully! Task ID: {result['task_id']}")
        else:
            st.error(f"Task submission failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
