"""
Comprehensive test runner for LLM-enhanced AWM trading system.
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Runs comprehensive tests for the LLM-enhanced trading system."""
    
    def __init__(self):
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "unit_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "paper_trading_validation": {},
            "overall_status": "PENDING"
        }
        
        self.required_services = [
            "llm-market-intelligence-server:8007",
            "signal-generation-server:8004",
            "decision-engine-server:8005",
            "market-data-server:8001"
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and validations."""
        logger.info("Starting comprehensive test suite for LLM-enhanced AWM system")
        
        try:
            # Step 1: Check system prerequisites
            await self._check_prerequisites()
            
            # Step 2: Run unit tests
            await self._run_unit_tests()
            
            # Step 3: Run integration tests
            await self._run_integration_tests()
            
            # Step 4: Run performance tests
            await self._run_performance_tests()
            
            # Step 5: Run paper trading validation
            await self._run_paper_trading_validation()
            
            # Step 6: Generate comprehensive report
            await self._generate_final_report()
            
            self.test_results["end_time"] = datetime.now().isoformat()
            self.test_results["overall_status"] = "COMPLETED"
            
            logger.info("Comprehensive test suite completed successfully")
            return self.test_results
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results["overall_status"] = "FAILED"
            self.test_results["error"] = str(e)
            raise
    
    async def _check_prerequisites(self):
        """Check system prerequisites and service availability."""
        logger.info("Checking system prerequisites...")
        
        prerequisites = {
            "python_version": sys.version,
            "required_packages": [],
            "service_availability": {},
            "gpu_availability": False,
            "redis_availability": False
        }
        
        # Check required packages
        required_packages = [
            "pytest", "torch", "transformers", "aiohttp", "aioredis", 
            "pandas", "numpy", "fastapi", "uvicorn"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                prerequisites["required_packages"].append({"package": package, "status": "AVAILABLE"})
            except ImportError:
                prerequisites["required_packages"].append({"package": package, "status": "MISSING"})
                logger.warning(f"Required package {package} is missing")
        
        # Check GPU availability
        try:
            import torch
            prerequisites["gpu_availability"] = torch.cuda.is_available()
            if prerequisites["gpu_availability"]:
                prerequisites["gpu_info"] = torch.cuda.get_device_name()
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
        
        # Check service availability
        for service in self.required_services:
            service_name, port = service.split(":")
            is_available = await self._check_service_health(f"http://localhost:{port}/health")
            prerequisites["service_availability"][service_name] = is_available
        
        self.test_results["prerequisites"] = prerequisites
        logger.info("Prerequisites check completed")
    
    async def _check_service_health(self, health_url: str) -> bool:
        """Check if a service is healthy."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _run_unit_tests(self):
        """Run unit tests for LLM components."""
        logger.info("Running unit tests...")
        
        unit_test_files = [
            "tests/test_llm_market_intelligence.py",
            "tests/test_llm_engine.py"  # If exists
        ]
        
        unit_results = {}
        
        for test_file in unit_test_files:
            if os.path.exists(test_file):
                logger.info(f"Running {test_file}")
                
                try:
                    # Run pytest for each test file
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                    ], capture_output=True, text=True, timeout=300)
                    
                    unit_results[test_file] = {
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "status": "PASSED" if result.returncode == 0 else "FAILED"
                    }
                    
                    if result.returncode == 0:
                        logger.info(f"✓ {test_file} passed")
                    else:
                        logger.error(f"✗ {test_file} failed")
                        
                except subprocess.TimeoutExpired:
                    unit_results[test_file] = {
                        "status": "TIMEOUT",
                        "error": "Test execution timed out"
                    }
                    logger.error(f"✗ {test_file} timed out")
                    
                except Exception as e:
                    unit_results[test_file] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                    logger.error(f"✗ {test_file} error: {e}")
            else:
                logger.warning(f"Test file {test_file} not found")
        
        self.test_results["unit_tests"] = unit_results
        logger.info("Unit tests completed")
    
    async def _run_integration_tests(self):
        """Run integration tests for LLM-enhanced components."""
        logger.info("Running integration tests...")
        
        integration_test_files = [
            "tests/test_llm_integration.py"
        ]
        
        integration_results = {}
        
        for test_file in integration_test_files:
            if os.path.exists(test_file):
                logger.info(f"Running {test_file}")
                
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                    ], capture_output=True, text=True, timeout=600)
                    
                    integration_results[test_file] = {
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "status": "PASSED" if result.returncode == 0 else "FAILED"
                    }
                    
                    if result.returncode == 0:
                        logger.info(f"✓ {test_file} passed")
                    else:
                        logger.error(f"✗ {test_file} failed")
                        
                except subprocess.TimeoutExpired:
                    integration_results[test_file] = {
                        "status": "TIMEOUT",
                        "error": "Test execution timed out"
                    }
                    
                except Exception as e:
                    integration_results[test_file] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        self.test_results["integration_tests"] = integration_results
        logger.info("Integration tests completed")
    
    async def _run_performance_tests(self):
        """Run performance validation tests."""
        logger.info("Running performance tests...")
        
        performance_test_files = [
            "tests/test_llm_performance.py"
        ]
        
        performance_results = {}
        
        for test_file in performance_test_files:
            if os.path.exists(test_file):
                logger.info(f"Running {test_file}")
                
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-s"
                    ], capture_output=True, text=True, timeout=900)
                    
                    performance_results[test_file] = {
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "status": "PASSED" if result.returncode == 0 else "FAILED"
                    }
                    
                    # Extract performance metrics from output
                    if "latency" in result.stdout.lower():
                        performance_results[test_file]["metrics_extracted"] = True
                    
                    if result.returncode == 0:
                        logger.info(f"✓ {test_file} passed")
                    else:
                        logger.error(f"✗ {test_file} failed")
                        
                except subprocess.TimeoutExpired:
                    performance_results[test_file] = {
                        "status": "TIMEOUT",
                        "error": "Performance test timed out"
                    }
                    
                except Exception as e:
                    performance_results[test_file] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        self.test_results["performance_tests"] = performance_results
        logger.info("Performance tests completed")
    
    async def _run_paper_trading_validation(self):
        """Run paper trading validation."""
        logger.info("Running paper trading validation...")
        
        try:
            # Import and run paper trading validation
            from scripts.paper_trading_validation import PaperTradingValidator
            
            validator = PaperTradingValidator()
            
            # Run shorter validation for testing (7 days instead of 30)
            validation_results = await validator.run_validation(days=7)
            
            self.test_results["paper_trading_validation"] = {
                "status": "COMPLETED",
                "results": validation_results,
                "summary": {
                    "llm_enhanced_return": validation_results.get("llm_enhanced_results", {}).get("total_return", 0),
                    "quantitative_only_return": validation_results.get("quantitative_only_results", {}).get("total_return", 0),
                    "improvement": validation_results.get("comparison_metrics", {}).get("return_improvement", 0),
                    "overall_assessment": validation_results.get("comparison_metrics", {}).get("overall_assessment", "UNKNOWN")
                }
            }
            
            logger.info("✓ Paper trading validation completed")
            
        except Exception as e:
            self.test_results["paper_trading_validation"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ Paper trading validation failed: {e}")
    
    async def _generate_final_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating final test report...")
        
        # Calculate overall statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_category in ["unit_tests", "integration_tests", "performance_tests"]:
            category_results = self.test_results.get(test_category, {})
            for test_file, result in category_results.items():
                total_tests += 1
                if result.get("status") == "PASSED":
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        # Add paper trading validation
        if self.test_results.get("paper_trading_validation", {}).get("status") == "COMPLETED":
            total_tests += 1
            passed_tests += 1
        else:
            total_tests += 1
            failed_tests += 1
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "recommendations": [],
            "next_steps": []
        }
        
        # Generate recommendations based on results
        if report["test_summary"]["success_rate"] >= 0.9:
            report["recommendations"].append("System is ready for production deployment")
        elif report["test_summary"]["success_rate"] >= 0.7:
            report["recommendations"].append("System shows good performance but needs minor fixes")
        else:
            report["recommendations"].append("System requires significant improvements before deployment")
        
        # Add specific recommendations
        paper_trading_result = self.test_results.get("paper_trading_validation", {})
        if paper_trading_result.get("status") == "COMPLETED":
            assessment = paper_trading_result.get("summary", {}).get("overall_assessment", "UNKNOWN")
            if assessment == "BETTER":
                report["recommendations"].append("LLM enhancement shows clear performance improvement")
            elif assessment == "MIXED":
                report["recommendations"].append("LLM enhancement shows mixed results - consider parameter tuning")
            else:
                report["recommendations"].append("LLM enhancement needs optimization")
        
        self.test_results["final_report"] = report
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("test_results", exist_ok=True)
        
        with open(f"test_results/comprehensive_test_results_{timestamp}.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info("Final test report generated")
    
    def print_summary(self):
        """Print test summary to console."""
        print("\n" + "="*80)
        print("LLM-ENHANCED AWM TRADING SYSTEM - COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        # Prerequisites
        print("\n📋 PREREQUISITES:")
        prereq = self.test_results.get("prerequisites", {})
        print(f"   GPU Available: {'✓' if prereq.get('gpu_availability') else '✗'}")
        
        services = prereq.get("service_availability", {})
        for service, available in services.items():
            print(f"   {service}: {'✓' if available else '✗'}")
        
        # Test Results
        print("\n🧪 TEST RESULTS:")
        for category in ["unit_tests", "integration_tests", "performance_tests"]:
            results = self.test_results.get(category, {})
            passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
            total = len(results)
            print(f"   {category.replace('_', ' ').title()}: {passed}/{total} passed")
        
        # Paper Trading
        paper_result = self.test_results.get("paper_trading_validation", {})
        if paper_result.get("status") == "COMPLETED":
            summary = paper_result.get("summary", {})
            improvement = summary.get("improvement", 0)
            assessment = summary.get("overall_assessment", "UNKNOWN")
            print(f"   Paper Trading: ✓ ({improvement:+.2%} improvement, {assessment})")
        else:
            print(f"   Paper Trading: ✗")
        
        # Final Assessment
        final_report = self.test_results.get("final_report", {})
        test_summary = final_report.get("test_summary", {})
        success_rate = test_summary.get("success_rate", 0)
        
        print(f"\n📊 OVERALL SUCCESS RATE: {success_rate:.1%}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in final_report.get("recommendations", []):
            print(f"   • {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main function to run comprehensive tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = ComprehensiveTestRunner()
    
    try:
        print("🚀 Starting Comprehensive Test Suite for LLM-Enhanced AWM Trading System")
        print("This may take 15-30 minutes to complete...")
        
        results = await runner.run_all_tests()
        runner.print_summary()
        
        return results
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        runner.print_summary()
        raise


if __name__ == "__main__":
    asyncio.run(main())
