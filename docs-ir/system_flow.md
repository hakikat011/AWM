# MFT-1 System Flow

## Normal Operation Flow

1. **Market Scanning Phase**
   - Meta-Agent polls MarketData_Server for instruments meeting criteria
   - Potential opportunities are queued for analysis

2. **Analysis Phase**
   - Analysis Agent performs deep-dive on each queued instrument
   - Queries multiple MCP servers for comprehensive data
   - Generates analysis report with recommendation

3. **Trade Proposal Phase**
   - Insight & Leverage Agent converts analysis to actionable trade
   - Calculates position size, entry/exit points, and risk parameters
   - Proposal is sent to Dashboard UI

4. **Approval Phase**
   - Operator reviews proposal on Dashboard
   - Approves or rejects based on judgment
   - Approved trades proceed to Risk Engine

5. **Execution Phase**
   - Risk Engine validates trade against safety parameters
   - OMS converts approved trade to broker-specific format
   - Order is placed and tracked until filled/rejected
   - Trade details logged to TradeLog_Server

## Recovery Flows

1. **Service Restart Recovery**
   - Each service must implement state recovery on restart
   - In-progress analyses are resumed from last checkpoint
   - Pending orders are reconciled with broker status

2. **Data Inconsistency Recovery**
   - Daily reconciliation job compares internal state with broker
   - Discrepancies trigger alerts and manual review process