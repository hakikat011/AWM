"""
Indian Market Order Types Support for AWM System.
Implements support for Indian market-specific order types including CNC, MIS, NRML, BO, CO.
"""

import logging
from datetime import datetime, time
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass

from .utils import validate_trading_hours, get_tick_size

logger = logging.getLogger(__name__)


class IndianProductType(Enum):
    """Indian market product types."""
    CNC = "CNC"  # Cash and Carry (Delivery)
    MIS = "MIS"  # Margin Intraday Square-off
    NRML = "NRML"  # Normal (F&O)


class IndianOrderType(Enum):
    """Indian market order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # Stop Loss
    SL_M = "SL-M"  # Stop Loss Market


class IndianOrderVariety(Enum):
    """Indian market order varieties."""
    REGULAR = "regular"
    BO = "bo"  # Bracket Order
    CO = "co"  # Cover Order
    AMO = "amo"  # After Market Order


@dataclass
class OrderValidationRule:
    """Order validation rule structure."""
    product_type: str
    order_type: str
    min_quantity: int
    max_quantity: int
    min_price: Optional[Decimal]
    max_price: Optional[Decimal]
    requires_trigger_price: bool
    requires_target_price: bool
    requires_stoploss_price: bool
    allowed_exchanges: List[str]
    allowed_segments: List[str]
    trading_hours_only: bool


class IndianOrderTypeManager:
    """
    Manager for Indian market-specific order types and validation.
    Handles CNC, MIS, NRML, BO, CO order types with proper validation.
    """
    
    def __init__(self):
        # Order type validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Product type configurations
        self.product_configs = {
            IndianProductType.CNC.value: {
                "name": "Cash and Carry (Delivery)",
                "description": "For delivery-based equity trading",
                "margin_required": False,
                "auto_square_off": False,
                "allowed_exchanges": ["NSE", "BSE"],
                "allowed_segments": ["EQ"],
                "settlement": "T+2"
            },
            IndianProductType.MIS.value: {
                "name": "Margin Intraday Square-off",
                "description": "For intraday trading with margin",
                "margin_required": True,
                "auto_square_off": True,
                "square_off_time": time(15, 20),  # 3:20 PM
                "allowed_exchanges": ["NSE", "BSE"],
                "allowed_segments": ["EQ", "FO"],
                "settlement": "Same day"
            },
            IndianProductType.NRML.value: {
                "name": "Normal (F&O)",
                "description": "For F&O trading",
                "margin_required": True,
                "auto_square_off": False,
                "allowed_exchanges": ["NFO", "BFO", "CDS", "MCX"],
                "allowed_segments": ["FO", "CD", "CO"],
                "settlement": "As per contract"
            }
        }
        
        # Special order type configurations
        self.special_order_configs = {
            IndianOrderVariety.BO.value: {
                "name": "Bracket Order",
                "description": "Order with automatic target and stop-loss",
                "requires_target": True,
                "requires_stoploss": True,
                "allowed_products": ["MIS"],
                "max_orders_per_day": 50,
                "min_profit_target": 0.5,  # Minimum 0.5% profit target
                "max_stoploss": 10.0  # Maximum 10% stop loss
            },
            IndianOrderVariety.CO.value: {
                "name": "Cover Order",
                "description": "Market order with compulsory stop-loss",
                "requires_target": False,
                "requires_stoploss": True,
                "allowed_products": ["MIS"],
                "max_orders_per_day": 100,
                "order_type_restriction": "MARKET"
            },
            IndianOrderVariety.AMO.value: {
                "name": "After Market Order",
                "description": "Orders placed outside market hours",
                "requires_target": False,
                "requires_stoploss": False,
                "allowed_products": ["CNC", "MIS", "NRML"],
                "allowed_time_range": {
                    "start": time(16, 0),  # 4:00 PM
                    "end": time(9, 15)     # 9:15 AM next day
                }
            }
        }
    
    def _initialize_validation_rules(self) -> Dict[str, OrderValidationRule]:
        """Initialize order validation rules."""
        rules = {}
        
        # CNC (Cash and Carry) rules
        rules["CNC_MARKET"] = OrderValidationRule(
            product_type="CNC",
            order_type="MARKET",
            min_quantity=1,
            max_quantity=10000,
            min_price=None,
            max_price=None,
            requires_trigger_price=False,
            requires_target_price=False,
            requires_stoploss_price=False,
            allowed_exchanges=["NSE", "BSE"],
            allowed_segments=["EQ"],
            trading_hours_only=True
        )
        
        rules["CNC_LIMIT"] = OrderValidationRule(
            product_type="CNC",
            order_type="LIMIT",
            min_quantity=1,
            max_quantity=10000,
            min_price=Decimal("0.01"),
            max_price=Decimal("100000"),
            requires_trigger_price=False,
            requires_target_price=False,
            requires_stoploss_price=False,
            allowed_exchanges=["NSE", "BSE"],
            allowed_segments=["EQ"],
            trading_hours_only=False
        )
        
        # MIS (Margin Intraday) rules
        rules["MIS_MARKET"] = OrderValidationRule(
            product_type="MIS",
            order_type="MARKET",
            min_quantity=1,
            max_quantity=50000,
            min_price=None,
            max_price=None,
            requires_trigger_price=False,
            requires_target_price=False,
            requires_stoploss_price=False,
            allowed_exchanges=["NSE", "BSE"],
            allowed_segments=["EQ"],
            trading_hours_only=True
        )
        
        rules["MIS_LIMIT"] = OrderValidationRule(
            product_type="MIS",
            order_type="LIMIT",
            min_quantity=1,
            max_quantity=50000,
            min_price=Decimal("0.01"),
            max_price=Decimal("100000"),
            requires_trigger_price=False,
            requires_target_price=False,
            requires_stoploss_price=False,
            allowed_exchanges=["NSE", "BSE"],
            allowed_segments=["EQ"],
            trading_hours_only=False
        )
        
        # NRML (F&O) rules
        rules["NRML_MARKET"] = OrderValidationRule(
            product_type="NRML",
            order_type="MARKET",
            min_quantity=1,
            max_quantity=100000,
            min_price=None,
            max_price=None,
            requires_trigger_price=False,
            requires_target_price=False,
            requires_stoploss_price=False,
            allowed_exchanges=["NFO", "BFO", "CDS", "MCX"],
            allowed_segments=["FO", "CD", "CO"],
            trading_hours_only=True
        )
        
        # Add more rules for other combinations...
        
        return rules
    
    def validate_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate order against Indian market rules.
        
        Args:
            order_request: Order details
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        try:
            product = order_request.get("product", "CNC").upper()
            order_type = order_request.get("order_type", "MARKET").upper()
            variety = order_request.get("variety", "regular").lower()
            
            # Basic product type validation
            if product not in [p.value for p in IndianProductType]:
                errors.append(f"Invalid product type: {product}")
                return {"valid": False, "errors": errors, "warnings": warnings}
            
            # Get validation rule
            rule_key = f"{product}_{order_type}"
            if rule_key not in self.validation_rules:
                errors.append(f"Unsupported order type {order_type} for product {product}")
                return {"valid": False, "errors": errors, "warnings": warnings}
            
            rule = self.validation_rules[rule_key]
            
            # Validate quantity
            quantity = int(order_request.get("quantity", 0))
            if quantity < rule.min_quantity:
                errors.append(f"Quantity {quantity} below minimum {rule.min_quantity}")
            elif quantity > rule.max_quantity:
                errors.append(f"Quantity {quantity} exceeds maximum {rule.max_quantity}")
            
            # Validate price
            if rule.min_price or rule.max_price:
                price = order_request.get("price")
                if price is not None:
                    price = Decimal(str(price))
                    if rule.min_price and price < rule.min_price:
                        errors.append(f"Price {price} below minimum {rule.min_price}")
                    elif rule.max_price and price > rule.max_price:
                        errors.append(f"Price {price} exceeds maximum {rule.max_price}")
            
            # Validate trading hours
            if rule.trading_hours_only:
                trading_status = validate_trading_hours()
                if not trading_status["is_trading_hours"]:
                    if variety != "amo":
                        errors.append("Order type requires trading hours or AMO variety")
            
            # Validate special order types
            if variety in ["bo", "co"]:
                special_validation = self._validate_special_order(order_request, variety)
                errors.extend(special_validation["errors"])
                warnings.extend(special_validation["warnings"])
            
            # Product-specific validations
            product_validation = self._validate_product_specific(order_request, product)
            errors.extend(product_validation["errors"])
            warnings.extend(product_validation["warnings"])
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "rule_applied": rule_key,
                "validated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": warnings
            }
    
    def _validate_special_order(self, order_request: Dict[str, Any], variety: str) -> Dict[str, Any]:
        """Validate special order types (BO, CO)."""
        errors = []
        warnings = []
        
        try:
            config = self.special_order_configs.get(variety, {})
            
            # Check allowed products
            product = order_request.get("product", "CNC").upper()
            if "allowed_products" in config:
                if product not in config["allowed_products"]:
                    errors.append(f"{variety.upper()} not allowed for product {product}")
            
            # Bracket Order validations
            if variety == "bo":
                if "target_price" not in order_request:
                    errors.append("Bracket Order requires target_price")
                if "stoploss_price" not in order_request:
                    errors.append("Bracket Order requires stoploss_price")
                
                # Validate profit/loss ratios
                price = order_request.get("price")
                target_price = order_request.get("target_price")
                stoploss_price = order_request.get("stoploss_price")
                
                if price and target_price and stoploss_price:
                    price = Decimal(str(price))
                    target_price = Decimal(str(target_price))
                    stoploss_price = Decimal(str(stoploss_price))
                    
                    side = order_request.get("side", "BUY").upper()
                    
                    if side == "BUY":
                        profit_pct = ((target_price - price) / price) * 100
                        loss_pct = ((price - stoploss_price) / price) * 100
                    else:
                        profit_pct = ((price - target_price) / price) * 100
                        loss_pct = ((stoploss_price - price) / price) * 100
                    
                    if profit_pct < config.get("min_profit_target", 0.5):
                        warnings.append(f"Profit target {profit_pct:.2f}% below recommended minimum")
                    
                    if loss_pct > config.get("max_stoploss", 10.0):
                        warnings.append(f"Stop loss {loss_pct:.2f}% above recommended maximum")
            
            # Cover Order validations
            elif variety == "co":
                if order_request.get("order_type", "MARKET").upper() != "MARKET":
                    errors.append("Cover Order must be MARKET order")
                if "stoploss_price" not in order_request:
                    errors.append("Cover Order requires stoploss_price")
            
            return {"errors": errors, "warnings": warnings}
            
        except Exception as e:
            logger.error(f"Error validating special order: {e}")
            return {"errors": [f"Special order validation error: {str(e)}"], "warnings": warnings}
    
    def _validate_product_specific(self, order_request: Dict[str, Any], product: str) -> Dict[str, Any]:
        """Validate product-specific rules."""
        errors = []
        warnings = []
        
        try:
            config = self.product_configs.get(product, {})
            
            # MIS specific validations
            if product == "MIS":
                # Check if order will be auto-squared off
                if config.get("auto_square_off"):
                    square_off_time = config.get("square_off_time")
                    if square_off_time:
                        warnings.append(f"MIS orders will be auto-squared off at {square_off_time}")
                
                # Check margin requirements
                if config.get("margin_required"):
                    warnings.append("MIS orders require sufficient margin")
            
            # CNC specific validations
            elif product == "CNC":
                # Check settlement
                settlement = config.get("settlement")
                if settlement:
                    warnings.append(f"CNC orders settle on {settlement}")
            
            # NRML specific validations
            elif product == "NRML":
                # Check if F&O segment
                symbol = order_request.get("symbol", "")
                if ":" in symbol:
                    exchange = symbol.split(":")[0]
                    if exchange not in config.get("allowed_exchanges", []):
                        errors.append(f"NRML not allowed for exchange {exchange}")
            
            return {"errors": errors, "warnings": warnings}
            
        except Exception as e:
            logger.error(f"Error in product-specific validation: {e}")
            return {"errors": [f"Product validation error: {str(e)}"], "warnings": warnings}
    
    def get_product_info(self, product: str) -> Dict[str, Any]:
        """Get product type information."""
        return self.product_configs.get(product.upper(), {})
    
    def get_supported_order_types(self, product: str) -> List[str]:
        """Get supported order types for a product."""
        supported = []
        for rule_key, rule in self.validation_rules.items():
            if rule.product_type == product.upper():
                supported.append(rule.order_type)
        return list(set(supported))
    
    def format_order_for_zerodha(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Format order request for Zerodha API."""
        try:
            # Basic mapping
            zerodha_order = {
                "variety": order_request.get("variety", "regular"),
                "exchange": self._extract_exchange(order_request.get("symbol", "")),
                "tradingsymbol": self._extract_tradingsymbol(order_request.get("symbol", "")),
                "transaction_type": order_request.get("side", "BUY").upper(),
                "quantity": int(order_request.get("quantity", 0)),
                "product": order_request.get("product", "CNC").upper(),
                "order_type": order_request.get("order_type", "MARKET").upper(),
                "validity": order_request.get("validity", "DAY").upper()
            }
            
            # Add price parameters
            if "price" in order_request:
                zerodha_order["price"] = float(order_request["price"])
            
            if "trigger_price" in order_request:
                zerodha_order["trigger_price"] = float(order_request["trigger_price"])
            
            # Add special order parameters
            variety = order_request.get("variety", "regular")
            if variety == "bo":
                if "target_price" in order_request:
                    zerodha_order["squareoff"] = float(order_request["target_price"])
                if "stoploss_price" in order_request:
                    zerodha_order["stoploss"] = float(order_request["stoploss_price"])
            
            elif variety == "co":
                if "stoploss_price" in order_request:
                    zerodha_order["trigger_price"] = float(order_request["stoploss_price"])
            
            return zerodha_order
            
        except Exception as e:
            logger.error(f"Error formatting order for Zerodha: {e}")
            raise
    
    def _extract_exchange(self, symbol: str) -> str:
        """Extract exchange from symbol."""
        if ":" in symbol:
            return symbol.split(":")[0]
        return "NSE"  # Default
    
    def _extract_tradingsymbol(self, symbol: str) -> str:
        """Extract trading symbol."""
        if ":" in symbol:
            return symbol.split(":")[1]
        return symbol


# Global instance
indian_order_manager = IndianOrderTypeManager()
