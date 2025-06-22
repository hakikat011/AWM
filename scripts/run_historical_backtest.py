"""
Run historical backtesting with configuration file support.
"""

import asyncio
import argparse
import logging
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.historical_backtesting import HistoricalBacktester


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def setup_logging(config: dict):
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/historical_backtesting.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class ConfigurableHistoricalBacktester(HistoricalBacktester):
    """Historical backtester with configuration file support."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._apply_configuration()
    
    def _apply_configuration(self):
        """Apply configuration to backtester."""
        # Backtest period
        backtest_config = self.config.get('backtest_period', {})
        if 'start_date' in backtest_config:
            self.backtest_start_date = datetime.fromisoformat(backtest_config['start_date'])
        if 'end_date' in backtest_config:
            self.backtest_end_date = datetime.fromisoformat(backtest_config['end_date'])
        
        # Portfolio configuration
        portfolio_config = self.config.get('portfolio', {})
        if 'initial_capital' in portfolio_config:
            self.initial_capital = portfolio_config['initial_capital']
        
        # Update portfolio instances
        self.llm_enhanced_portfolio["cash"] = self.initial_capital
        self.quantitative_only_portfolio["cash"] = self.initial_capital
        
        # Symbols
        symbols_config = self.config.get('symbols', [])
        if symbols_config:
            self.test_symbols = symbols_config
        
        # Transaction costs
        transaction_costs = self.config.get('transaction_costs', {})
        if transaction_costs:
            self.transaction_costs.update(transaction_costs)
        
        # Slippage configuration
        slippage_config = self.config.get('slippage', {})
        if slippage_config:
            self.slippage_config.update(slippage_config)
        
        # Server URLs
        servers_config = self.config.get('servers', {})
        if servers_config:
            for key, url in servers_config.items():
                if key in self.server_urls:
                    self.server_urls[key] = url
    
    async def run_configured_backtest(self) -> dict:
        """Run backtesting with applied configuration."""
        logging.info("Starting configured historical backtesting...")
        logging.info(f"Configuration applied:")
        logging.info(f"  Period: {self.backtest_start_date} to {self.backtest_end_date}")
        logging.info(f"  Capital: ₹{self.initial_capital:,.0f}")
        logging.info(f"  Symbols: {', '.join(self.test_symbols)}")
        
        return await self.run_historical_backtest(
            start_date=self.backtest_start_date,
            end_date=self.backtest_end_date
        )


def print_configuration_summary(config: dict):
    """Print configuration summary."""
    print("\n" + "="*80)
    print("HISTORICAL BACKTESTING CONFIGURATION")
    print("="*80)
    
    # Backtest period
    backtest_period = config.get('backtest_period', {})
    print(f"Period: {backtest_period.get('start_date', 'N/A')} to {backtest_period.get('end_date', 'N/A')}")
    
    # Portfolio
    portfolio = config.get('portfolio', {})
    print(f"Initial Capital: ₹{portfolio.get('initial_capital', 0):,.0f}")
    print(f"Max Position Size: {portfolio.get('max_position_size_pct', 0):.1%}")
    
    # Symbols
    symbols = config.get('symbols', [])
    print(f"Symbols: {', '.join(symbols)}")
    
    # Transaction costs
    transaction_costs = config.get('transaction_costs', {})
    print(f"Total Transaction Cost: {transaction_costs.get('total_cost_pct', 0):.3%}")
    
    # Production readiness criteria
    prod_criteria = config.get('production_readiness', {})
    perf_criteria = prod_criteria.get('performance_criteria', {})
    print(f"Min Return Improvement: {perf_criteria.get('min_return_improvement', 0):.1%}")
    print(f"Max Drawdown Tolerance: {perf_criteria.get('max_drawdown_tolerance', 0):.1%}")
    
    print("="*80)


def print_results_summary(results: dict):
    """Print results summary."""
    print("\n" + "="*80)
    print("BACKTESTING RESULTS SUMMARY")
    print("="*80)
    
    llm_results = results.get("llm_enhanced_results", {})
    quant_results = results.get("quantitative_only_results", {})
    production_readiness = results.get("production_readiness", {})
    statistical_tests = results.get("statistical_tests", {})
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'LLM-Enhanced':<15} {'Quantitative':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Total Return':<25} {llm_results.get('total_return', 0):<15.2%} {quant_results.get('total_return', 0):<15.2%} {(llm_results.get('total_return', 0) - quant_results.get('total_return', 0)):<15.2%}")
    print(f"{'Sharpe Ratio':<25} {llm_results.get('sharpe_ratio', 0):<15.3f} {quant_results.get('sharpe_ratio', 0):<15.3f} {(llm_results.get('sharpe_ratio', 0) - quant_results.get('sharpe_ratio', 0)):<15.3f}")
    print(f"{'Max Drawdown':<25} {llm_results.get('max_drawdown', 0):<15.2%} {quant_results.get('max_drawdown', 0):<15.2%} {(quant_results.get('max_drawdown', 0) - llm_results.get('max_drawdown', 0)):<15.2%}")
    print(f"{'Win Rate':<25} {llm_results.get('win_rate', 0):<15.2%} {quant_results.get('win_rate', 0):<15.2%} {(llm_results.get('win_rate', 0) - quant_results.get('win_rate', 0)):<15.2%}")
    print(f"{'Total Trades':<25} {llm_results.get('total_trades', 0):<15} {quant_results.get('total_trades', 0):<15} {(llm_results.get('total_trades', 0) - quant_results.get('total_trades', 0)):<15}")
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    overall_sig = statistical_tests.get('overall_significance', {})
    print(f"Tests Significant: {overall_sig.get('tests_significant', 0)}/{overall_sig.get('total_tests', 0)}")
    print(f"Conclusion: {overall_sig.get('conclusion', 'N/A')}")
    
    print(f"\n🎯 PRODUCTION READINESS:")
    print(f"Overall Score: {production_readiness.get('overall_readiness_score', 0):.1%}")
    print(f"Recommendation: {production_readiness.get('deployment_recommendation', 'N/A')}")
    
    # Deployment plan
    deployment_plan = production_readiness.get('deployment_plan', {})
    print(f"\n📋 DEPLOYMENT PLAN:")
    print(f"Initial Allocation: {deployment_plan.get('initial_allocation', 'N/A')}")
    print(f"Monitoring Frequency: {deployment_plan.get('monitoring_frequency', 'N/A')}")
    
    # Success criteria
    success_criteria = deployment_plan.get('success_criteria', {})
    print(f"\n✅ SUCCESS CRITERIA:")
    for criterion, value in success_criteria.items():
        print(f"  {criterion}: {value}")
    
    print("="*80)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run historical backtesting for LLM-enhanced AWM trading system")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/backtesting_config.yaml",
        help="Path to configuration file (default: config/backtesting_config.yaml)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Override start date (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Override end date (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Override symbols to test"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        help="Override initial capital amount"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without running backtest"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(config)
        
        # Apply command line overrides
        if args.start_date:
            config['backtest_period']['start_date'] = args.start_date
        if args.end_date:
            config['backtest_period']['end_date'] = args.end_date
        if args.symbols:
            config['symbols'] = args.symbols
        if args.initial_capital:
            config['portfolio']['initial_capital'] = args.initial_capital
        
        # Print configuration
        print_configuration_summary(config)
        
        if args.dry_run:
            print("\n🔍 Dry run mode - configuration validated successfully")
            return
        
        # Confirm execution
        response = input("\nProceed with backtesting? (y/N): ")
        if response.lower() != 'y':
            print("Backtesting cancelled")
            return
        
        # Create and run backtester
        backtester = ConfigurableHistoricalBacktester(config)
        
        print("\n🚀 Starting historical backtesting...")
        start_time = datetime.now()
        
        results = await backtester.run_configured_backtest()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ Backtesting completed in {duration}")
        
        # Print results summary
        print_results_summary(results)
        
        # Save additional configuration-specific results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_results_file = f"test_results/backtest_config_{timestamp}.yaml"
        
        with open(config_results_file, 'w') as f:
            yaml.dump({
                'configuration_used': config,
                'execution_info': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'command_line_args': vars(args)
                }
            }, f, indent=2)
        
        print(f"\n📄 Configuration and execution info saved to: {config_results_file}")
        print("📊 Detailed results available in test_results/ directory")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Backtesting interrupted by user")
        return None
    except Exception as e:
        logging.error(f"Backtesting failed: {e}")
        print(f"\n❌ Backtesting failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
