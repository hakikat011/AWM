"""
Risk Assessment Agent for AWM system.
Evaluates portfolio risk and provides risk-adjusted position sizing recommendations.
"""

import asyncio
import logging
import json
import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
import numpy as np
import openai

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.agents.base_agent import BaseAgent
from shared.models.trading import RiskLevel

logger = logging.getLogger(__name__)


class RiskAssessmentAgent(BaseAgent):
    """Agent responsible for portfolio risk assessment and position sizing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("risk_assessment_agent", config)
        
        # OpenAI configuration
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Risk parameters from environment
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "100000"))
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "10000"))
        self.max_portfolio_risk = float(os.getenv("MAX_PORTFOLIO_RISK", "0.02"))
        self.default_stop_loss = float(os.getenv("DEFAULT_STOP_LOSS", "0.05"))
        
        # Risk calculation parameters
        self.confidence_level = self.config.get("confidence_level", 0.95)
        self.lookback_days = self.config.get("lookback_days", 252)  # 1 year
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)
    
    async def initialize(self):
        """Initialize the Risk Assessment Agent."""
        self.logger.info("Initializing Risk Assessment Agent...")
        
        # Test connections to required MCP servers
        try:
            # Test risk assessment server
            await self.call_mcp_server("risk_assessment", "health", {})
            self.logger.info("✓ Risk Assessment Server connection verified")
            
            # Test portfolio management server
            await self.call_mcp_server("portfolio_management", "health", {})
            self.logger.info("✓ Portfolio Management Server connection verified")
            
            # Test market data server
            await self.call_mcp_server("market_data", "health", {})
            self.logger.info("✓ Market Data Server connection verified")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to required servers: {e}")
            raise
        
        self.logger.info("Risk Assessment Agent initialized successfully")
    
    async def cleanup(self):
        """Cleanup the Risk Assessment Agent."""
        self.logger.info("Cleaning up Risk Assessment Agent...")
        # No specific cleanup needed
    
    async def process_task(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a risk assessment task."""
        
        if task_type == "assess_portfolio_risk":
            return await self._assess_portfolio_risk(parameters)
        elif task_type == "calculate_position_size":
            return await self._calculate_position_size(parameters)
        elif task_type == "evaluate_trade_risk":
            return await self._evaluate_trade_risk(parameters)
        elif task_type == "generate_risk_report":
            return await self._generate_risk_report(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _assess_portfolio_risk(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk."""
        portfolio_id = parameters["portfolio_id"]
        
        self.logger.info(f"Assessing risk for portfolio: {portfolio_id}")
        
        try:
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(portfolio_data)
            
            # Assess concentration risk
            concentration_risk = await self._assess_concentration_risk(portfolio_data)
            
            # Calculate correlation risk
            correlation_risk = await self._assess_correlation_risk(portfolio_data)
            
            # Generate AI-powered risk assessment
            ai_risk_assessment = await self._generate_ai_risk_assessment(
                portfolio_data, risk_metrics, concentration_risk, correlation_risk
            )
            
            # Combine all assessments
            result = {
                "portfolio_id": portfolio_id,
                "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_metrics": risk_metrics,
                "concentration_risk": concentration_risk,
                "correlation_risk": correlation_risk,
                "ai_assessment": ai_risk_assessment,
                "overall_risk_level": ai_risk_assessment.get("risk_level"),
                "recommendations": ai_risk_assessment.get("recommendations", [])
            }
            
            # Store risk assessment
            await self._store_risk_assessment(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            raise
    
    async def _get_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio data."""
        # Get portfolio details
        portfolio_response = await self.call_mcp_server(
            "portfolio_management",
            "get_portfolio",
            {"portfolio_id": portfolio_id}
        )
        
        # Get portfolio positions
        positions_response = await self.call_mcp_server(
            "portfolio_management",
            "get_positions",
            {"portfolio_id": portfolio_id}
        )
        
        # Get historical performance
        performance_response = await self.call_mcp_server(
            "portfolio_management",
            "get_performance",
            {
                "portfolio_id": portfolio_id,
                "days": self.lookback_days
            }
        )
        
        return {
            "portfolio": portfolio_response.get("portfolio", {}),
            "positions": positions_response.get("positions", []),
            "performance": performance_response.get("performance", [])
        }
    
    async def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        portfolio = portfolio_data["portfolio"]
        positions = portfolio_data["positions"]
        performance = portfolio_data["performance"]
        
        if not performance:
            return {"error": "No performance data available"}
        
        # Calculate returns
        returns = []
        for i in range(1, len(performance)):
            prev_value = float(performance[i-1]["portfolio_value"])
            curr_value = float(performance[i]["portfolio_value"])
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        if not returns:
            return {"error": "Insufficient data for risk calculation"}
        
        returns_array = np.array(returns)
        
        # Calculate basic risk metrics
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        mean_return = np.mean(returns_array) * 252  # Annualized
        
        # Value at Risk (VaR)
        var_1d = np.percentile(returns_array, (1 - self.confidence_level) * 100)
        var_5d = var_1d * np.sqrt(5)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = np.mean(returns_array[returns_array <= var_1d])
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sharpe Ratio (assuming risk-free rate of 5%)
        risk_free_rate = 0.05
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Portfolio value and risk amounts
        portfolio_value = float(portfolio.get("current_value", 0))
        var_amount_1d = abs(var_1d * portfolio_value)
        var_amount_5d = abs(var_5d * portfolio_value)
        
        return {
            "volatility": float(volatility),
            "var_1d": float(var_1d),
            "var_5d": float(var_5d),
            "var_amount_1d": float(var_amount_1d),
            "var_amount_5d": float(var_amount_5d),
            "expected_shortfall": float(expected_shortfall),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
            "mean_return": float(mean_return),
            "portfolio_value": portfolio_value,
            "data_points": len(returns)
        }
    
    async def _assess_concentration_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio concentration risk."""
        positions = portfolio_data["positions"]
        portfolio_value = float(portfolio_data["portfolio"].get("current_value", 0))
        
        if portfolio_value == 0:
            return {"error": "Portfolio value is zero"}
        
        # Calculate position weights
        position_weights = []
        sector_weights = {}
        
        for position in positions:
            weight = float(position.get("market_value", 0)) / portfolio_value
            position_weights.append(weight)
            
            # Group by sector (simplified - would need instrument metadata)
            sector = position.get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # Calculate concentration metrics
        herfindahl_index = sum(w**2 for w in position_weights)
        max_position_weight = max(position_weights) if position_weights else 0
        top_5_concentration = sum(sorted(position_weights, reverse=True)[:5])
        
        # Assess risk level
        if max_position_weight > 0.3 or herfindahl_index > 0.2:
            risk_level = "HIGH"
        elif max_position_weight > 0.15 or herfindahl_index > 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "herfindahl_index": float(herfindahl_index),
            "max_position_weight": float(max_position_weight),
            "top_5_concentration": float(top_5_concentration),
            "sector_weights": {k: float(v) for k, v in sector_weights.items()},
            "risk_level": risk_level,
            "number_of_positions": len(positions)
        }
    
    async def _assess_correlation_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess correlation risk between portfolio positions."""
        positions = portfolio_data["positions"]
        
        if len(positions) < 2:
            return {"risk_level": "LOW", "message": "Insufficient positions for correlation analysis"}
        
        # Get correlation data for major positions
        correlations = []
        high_correlation_pairs = []
        
        # Simplified correlation assessment
        # In a real implementation, you would calculate actual correlations
        # using historical price data for each instrument
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:], i+1):
                # Simplified correlation based on sector/exchange
                correlation = self._estimate_correlation(pos1, pos2)
                correlations.append(correlation)
                
                if correlation > self.correlation_threshold:
                    high_correlation_pairs.append({
                        "instrument1": pos1.get("symbol"),
                        "instrument2": pos2.get("symbol"),
                        "correlation": correlation
                    })
        
        avg_correlation = np.mean(correlations) if correlations else 0
        max_correlation = max(correlations) if correlations else 0
        
        # Assess risk level
        if max_correlation > 0.8 or len(high_correlation_pairs) > len(positions) * 0.3:
            risk_level = "HIGH"
        elif max_correlation > 0.6 or len(high_correlation_pairs) > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "average_correlation": float(avg_correlation),
            "max_correlation": float(max_correlation),
            "high_correlation_pairs": high_correlation_pairs,
            "risk_level": risk_level
        }
    
    def _estimate_correlation(self, pos1: Dict[str, Any], pos2: Dict[str, Any]) -> float:
        """Estimate correlation between two positions (simplified)."""
        # This is a simplified estimation
        # In practice, you would calculate actual correlation using price data
        
        # Same sector = higher correlation
        if pos1.get("sector") == pos2.get("sector"):
            return 0.7
        
        # Same exchange = medium correlation
        if pos1.get("exchange") == pos2.get("exchange"):
            return 0.4
        
        # Different sectors/exchanges = lower correlation
        return 0.2
    
    async def _generate_ai_risk_assessment(
        self,
        portfolio_data: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        concentration_risk: Dict[str, Any],
        correlation_risk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI-powered risk assessment."""
        
        context = {
            "portfolio_value": portfolio_data["portfolio"].get("current_value"),
            "number_of_positions": len(portfolio_data["positions"]),
            "risk_metrics": risk_metrics,
            "concentration_risk": concentration_risk,
            "correlation_risk": correlation_risk
        }
        
        prompt = f"""
        As a risk management expert, analyze the following portfolio risk data and provide a comprehensive assessment.
        
        Portfolio Data:
        {json.dumps(context, indent=2)}
        
        Please provide your assessment in the following JSON format:
        {{
            "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
            "overall_score": 0-100,
            "key_risks": ["list", "of", "main", "risks"],
            "recommendations": ["list", "of", "specific", "recommendations"],
            "position_sizing_advice": "guidance on position sizing",
            "diversification_advice": "guidance on diversification",
            "immediate_actions": ["urgent", "actions", "if", "any"]
        }}
        
        Consider:
        1. Portfolio volatility and VaR
        2. Concentration risk
        3. Correlation risk
        4. Maximum drawdown
        5. Sharpe ratio
        6. Overall risk-adjusted returns
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert risk management analyst specializing in portfolio risk assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                json_str = ai_response[start_idx:end_idx]
                
                assessment = json.loads(json_str)
                
                # Validate response
                assessment["risk_level"] = assessment.get("risk_level", "MEDIUM").upper()
                if assessment["risk_level"] not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                    assessment["risk_level"] = "MEDIUM"
                
                assessment["overall_score"] = max(0, min(100, int(assessment.get("overall_score", 50))))
                
                return assessment
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse AI risk assessment: {e}")
                return {
                    "risk_level": "MEDIUM",
                    "overall_score": 50,
                    "recommendations": ["AI analysis failed - manual review required"],
                    "error": str(e)
                }
                
        except Exception as e:
            self.logger.error(f"Error generating AI risk assessment: {e}")
            return {
                "risk_level": "MEDIUM",
                "overall_score": 50,
                "recommendations": ["Risk assessment failed - manual review required"],
                "error": str(e)
            }
    
    async def _calculate_position_size(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position size for a trade."""
        portfolio_id = parameters["portfolio_id"]
        symbol = parameters["symbol"]
        entry_price = float(parameters["entry_price"])
        stop_loss = float(parameters.get("stop_loss", entry_price * (1 - self.default_stop_loss)))
        risk_per_trade = float(parameters.get("risk_per_trade", 0.01))  # 1% default
        
        try:
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            portfolio_value = float(portfolio_data["portfolio"].get("current_value", 0))
            
            # Calculate risk amount
            risk_amount = portfolio_value * risk_per_trade
            
            # Calculate position size based on stop loss
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                raise ValueError("Stop loss cannot equal entry price")
            
            shares = int(risk_amount / price_risk)
            position_value = shares * entry_price
            
            # Apply position size limits
            max_position_value = min(
                self.max_position_size,
                portfolio_value * 0.2  # Max 20% of portfolio
            )
            
            if position_value > max_position_value:
                shares = int(max_position_value / entry_price)
                position_value = shares * entry_price
                actual_risk = shares * price_risk
            else:
                actual_risk = risk_amount
            
            return {
                "symbol": symbol,
                "recommended_shares": shares,
                "position_value": position_value,
                "risk_amount": actual_risk,
                "risk_percentage": actual_risk / portfolio_value,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "portfolio_allocation": position_value / portfolio_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            raise
    
    async def _evaluate_trade_risk(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk for a specific trade."""
        # Implementation for trade-specific risk evaluation
        pass
    
    async def _generate_risk_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        # Implementation for detailed risk reporting
        pass
    
    async def _store_risk_assessment(self, assessment: Dict[str, Any]) -> None:
        """Store risk assessment in database."""
        try:
            query = """
                INSERT INTO risk_metrics 
                (time, portfolio_id, var_1d, var_5d, expected_shortfall, sharpe_ratio, 
                 max_drawdown, volatility, risk_level)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            risk_metrics = assessment.get("risk_metrics", {})
            
            await db_manager.execute_query(
                query,
                datetime.now(timezone.utc),
                assessment["portfolio_id"],
                risk_metrics.get("var_1d"),
                risk_metrics.get("var_5d"),
                risk_metrics.get("expected_shortfall"),
                risk_metrics.get("sharpe_ratio"),
                risk_metrics.get("max_drawdown"),
                risk_metrics.get("volatility"),
                assessment.get("overall_risk_level", "MEDIUM")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store risk assessment: {e}")


async def main():
    """Main function to run the Risk Assessment Agent."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start agent
    agent = RiskAssessmentAgent()
    
    try:
        await agent.start()
        
        # Keep the agent running
        while agent.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
