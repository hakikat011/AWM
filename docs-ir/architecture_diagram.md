q# MFT-1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Layer 4: Control & Oversight                 │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Dashboard UI  │  │ Alerting Service│  │   KILL SWITCH   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└──────────┬────────────────────┬────────────────────┬────────────┘
           │                    │                    │             
           ▼                    ▼                    ▼             
┌─────────────────────────────────────────────────────────────────┐
│                     Layer 3: Execution & Risk                    │
│                                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │  Risk Management Engine │    │ Order Management System(OMS) │ │
│  └─────────────┬───────────┘    └──────────────┬──────────────┘ │
└────────────────┼────────────────────────────────────────────────┘
                 │                               │                 
                 ▼                               ▼                 
┌─────────────────────────────────────────────────────────────────┐
│                     Layer 2: Intelligence                        │
│                                                                  │
│  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────────┐│
│  │  Meta-Agent │  │  Analysis Agent │  │Insight & Leverage Agent││
│  └──────┬──────┘  └────────┬────────┘  └───────────┬───────────┘│
└─────────┼─────────────────┼─────────────────────────────────────┘
          │                 │                         │            
          ▼                 ▼                         ▼            
┌─────────────────────────────────────────────────────────────────┐
│                     Layer 1: Data & Tooling                      │
│                                                                  │
│ ┌────────────┐ ┌────────────┐ ┌─────────┐ ┌────────┐ ┌─────────┐│
│ │MarketData_ │ │TechnicalAna│ │ News_   │ │Zerodha │ │TradeLog_││
│ │  Server    │ │lysis_Server│ │ Server  │ │API_Serv│ │ Server  ││
│ └────────────┘ └────────────┘ └─────────┘ └────────┘ └─────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Communication Flow

All components communicate using the Model Context Protocol (MCP), which standardizes:
- Request/response format
- Error handling
- Authentication
- Rate limiting

The Meta-Agent orchestrates the workflow by coordinating the other agents and services.