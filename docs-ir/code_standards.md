# MFT-1 Code Standards

## Coding Style
- Follow PEP 8 for Python code
- Use type hints for all function parameters and return values
- Maximum line length: 100 characters
- Use descriptive variable names (no abbreviations except for common ones)

## Error Handling
1. **Hierarchical Error Types**:
   - Base exception class: `MFTError`
   - Layer-specific exceptions: `DataLayerError`, `IntelligenceError`, etc.
   - Component-specific exceptions: `MarketDataError`, `AnalysisAgentError`

2. **Error Logging**:
   - All exceptions must be logged with appropriate severity
   - Include context data with errors (e.g., instrument ID, timestamp)
   - Critical errors must trigger alerts via the Alerting Service

3. **Graceful Degradation**:
   - Services should attempt to continue operation when possible
   - The Risk Engine must halt trading on critical errors

## Testing Requirements
- Minimum 85% code coverage for all components
- Mock external APIs in tests
- Include performance tests for data-intensive operations