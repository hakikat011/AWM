#!/usr/bin/env python3
"""
Test script to verify AWM system setup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.mcp_client.base import MCPClient, MCPMessage, MessageType


async def test_mcp_communication():
    """Test basic MCP communication."""
    print("Testing MCP communication...")
    
    try:
        async with MCPClient("test_client") as client:
            # Test creating a message
            message = client._create_request_message(
                "test_server",
                {"test": "data"}
            )
            
            print(f"✓ Created MCP message: {message.request_id}")
            print(f"✓ Message type: {message.message_type.value}")
            print(f"✓ Source: {message.source}")
            print(f"✓ Destination: {message.destination}")
            
            # Test message serialization
            message_dict = message.to_dict()
            reconstructed = MCPMessage.from_dict(message_dict)
            
            assert reconstructed.request_id == message.request_id
            assert reconstructed.source == message.source
            print("✓ Message serialization/deserialization works")
            
    except Exception as e:
        print(f"✗ MCP communication test failed: {e}")
        return False
    
    return True


def test_project_structure():
    """Test that project structure is correct."""
    print("Testing project structure...")
    
    required_dirs = [
        "services/mcp_servers/market_data",
        "services/mcp_servers/technical_analysis",
        "services/agents",
        "services/execution",
        "shared/mcp_client",
        "shared/database",
        "shared/models",
        "database/init",
        "tests/unit"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - missing")
            return False
    
    return True


def test_required_files():
    """Test that required files exist."""
    print("Testing required files...")
    
    required_files = [
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "README.md",
        "database/init/01_create_extensions.sql",
        "database/init/02_create_tables.sql",
        "database/init/03_initial_data.sql",
        "shared/mcp_client/base.py",
        "shared/mcp_client/server.py",
        "shared/database/connection.py",
        "shared/models/trading.py"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - missing")
            return False
    
    return True


def test_environment_template():
    """Test that .env.example has required variables."""
    print("Testing environment template...")
    
    env_file = project_root / ".env.example"
    if not env_file.exists():
        print("✗ .env.example file missing")
        return False
    
    content = env_file.read_text()
    
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "ZERODHA_API_KEY",
        "ZERODHA_API_SECRET",
        "OPENAI_API_KEY",
        "MAX_POSITION_SIZE",
        "MAX_DAILY_LOSS",
        "TELEGRAM_BOT_TOKEN"
    ]
    
    for var in required_vars:
        if var in content:
            print(f"✓ {var}")
        else:
            print(f"✗ {var} - missing from .env.example")
            return False
    
    return True


def test_docker_compose():
    """Test docker-compose.yml structure."""
    print("Testing Docker Compose configuration...")
    
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        print("✗ docker-compose.yml missing")
        return False
    
    content = compose_file.read_text()
    
    required_services = [
        "timescaledb",
        "redis",
        "market-data-server",
        "technical-analysis-server",
        "dashboard"
    ]
    
    for service in required_services:
        if service in content:
            print(f"✓ {service} service defined")
        else:
            print(f"✗ {service} service missing")
            return False
    
    return True


async def main():
    """Run all tests."""
    print("=" * 50)
    print("AWM System Setup Test")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Required Files", test_required_files),
        ("Environment Template", test_environment_template),
        ("Docker Compose", test_docker_compose),
        ("MCP Communication", test_mcp_communication)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your AWM system setup is ready.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your API keys")
        print("2. Run: docker-compose build")
        print("3. Run: docker-compose up -d")
        print("4. Access dashboard at http://localhost:8501")
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
