"""
Settlement - Handles trade settlement and cash management
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from utils.logger import logger as logging

class Settlement:
    """
    Settlement - Handles trade settlement and cash management
    
    Responsibilities:
    - Track pending settlements
    - Manage cash balances
    - Calculate buying power
    - Handle corporate actions
    - Generate settlement reports
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Account balance
        self.cash_balance = config.get("initial_cash", 100000.0)
        self.initial_cash = self.cash_balance
        
        # Settlement tracking
        self.pending_settlements = []  # T+2 settlements
        self.settlement_days = config.get("settlement_days", 2)  # T+2
        
        # Transaction history
        self.transactions = []
        self.max_transactions = config.get("max_transactions", 10000)
        
        # Margin parameters
        self.margin_enabled = config.get("margin_enabled", False)
        self.margin_multiplier = config.get("margin_multiplier", 2.0)
        self.maintenance_margin = config.get("maintenance_margin", 0.25)  # 25% maintenance
        
        # Corporate actions tracking
        self.corporate_actions = []
        
        logging.info(f"✅ Settlement initialized with ${self.cash_balance:,.2f}")
    
    def process_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trade for settlement
        """
        symbol = trade["symbol"]
        quantity = trade["quantity"]
        price = trade["price"]
        side = trade["side"]
        trade_id = trade.get("order_id", f"trade_{datetime.now().timestamp()}")
        
        trade_value = quantity * price
        
        # Calculate settlement date
        settlement_date = datetime.now() + timedelta(days=self.settlement_days)
        
        if side == "BUY":
            # Check if enough cash (including margin)
            available = self.get_available_cash()
            if trade_value > available and not self.margin_enabled:
                return {
                    "success": False,
                    "error": "Insufficient funds",
                    "available": available,
                    "required": trade_value,
                    "shortfall": trade_value - available
                }
            
            # Record pending settlement
            settlement = {
                "id": f"settle_{datetime.now().timestamp()}",
                "trade_id": trade_id,
                "type": "debit",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "amount": trade_value,
                "trade_date": datetime.now().isoformat(),
                "settlement_date": settlement_date.isoformat(),
                "status": "pending"
            }
            self.pending_settlements.append(settlement)
            
            # Immediate cash impact (for margin accounts, this is when cash is reserved)
            if not self.margin_enabled:
                self.cash_balance -= trade_value
                cash_impact = -trade_value
            else:
                # With margin, we reserve the cash but it's not deducted yet
                cash_impact = 0
            
        else:  # SELL
            # Record pending settlement
            settlement = {
                "id": f"settle_{datetime.now().timestamp()}",
                "trade_id": trade_id,
                "type": "credit",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "amount": trade_value,
                "trade_date": datetime.now().isoformat(),
                "settlement_date": settlement_date.isoformat(),
                "status": "pending"
            }
            self.pending_settlements.append(settlement)
            
            # Immediate cash impact (cash is available immediately for trading)
            self.cash_balance += trade_value
            cash_impact = trade_value
        
        # Record transaction
        self._add_transaction({
            "timestamp": datetime.now().isoformat(),
            "type": "trade",
            "trade_id": trade_id,
            "side": side,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": trade_value,
            "cash_impact": cash_impact,
            "settlement_date": settlement_date.isoformat()
        })
        
        logging.info(f"💰 Trade processed: {side} {quantity} {symbol} @ ${price:.2f}")
        
        return {
            "success": True,
            "trade_id": trade_id,
            "settlement_id": settlement["id"],
            "cash_balance": self.cash_balance,
            "pending_settlements": len(self.pending_settlements),
            "settlement_date": settlement_date.isoformat(),
            "trade_value": trade_value,
            "cash_impact": cash_impact
        }
    
    def process_dividend(self, symbol: str, amount_per_share: float, 
                        shares: int, ex_date: str = None) -> Dict[str, Any]:
        """
        Process a dividend payment
        """
        total_amount = amount_per_share * shares
        payment_date = datetime.now()
        
        if ex_date:
            try:
                ex_date_obj = datetime.fromisoformat(ex_date)
                if payment_date < ex_date_obj:
                    return {
                        "success": False,
                        "error": "Dividend not yet payable",
                        "payable_date": ex_date
                    }
            except:
                pass
        
        # Cash is available immediately
        self.cash_balance += total_amount
        
        # Record transaction
        self._add_transaction({
            "timestamp": datetime.now().isoformat(),
            "type": "dividend",
            "symbol": symbol,
            "shares": shares,
            "amount_per_share": amount_per_share,
            "total_amount": total_amount,
            "cash_impact": total_amount
        })
        
        logging.info(f"💰 Dividend received: ${total_amount:,.2f} from {symbol}")
        
        return {
            "success": True,
            "symbol": symbol,
            "amount": total_amount,
            "cash_balance": self.cash_balance
        }
    
    def process_interest(self, amount: float, description: str = "Interest") -> Dict[str, Any]:
        """
        Process interest payment (positive) or charge (negative)
        """
        self.cash_balance += amount
        
        self._add_transaction({
            "timestamp": datetime.now().isoformat(),
            "type": "interest",
            "description": description,
            "amount": amount,
            "cash_impact": amount
        })
        
        if amount > 0:
            logging.info(f"💰 Interest received: ${amount:,.2f}")
        else:
            logging.info(f"💸 Interest charged: ${abs(amount):,.2f}")
        
        return {
            "success": True,
            "amount": amount,
            "cash_balance": self.cash_balance
        }
    
    def process_fee(self, amount: float, description: str = "Fee") -> Dict[str, Any]:
        """
        Process a fee (commission, etc.)
        """
        self.cash_balance -= abs(amount)
        
        self._add_transaction({
            "timestamp": datetime.now().isoformat(),
            "type": "fee",
            "description": description,
            "amount": -abs(amount),
            "cash_impact": -abs(amount)
        })
        
        logging.info(f"💸 Fee charged: ${abs(amount):,.2f} - {description}")
        
        return {
            "success": True,
            "amount": -abs(amount),
            "cash_balance": self.cash_balance
        }
    
    def process_corporate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a corporate action (split, merger, etc.)
        """
        action_type = action.get("type")
        symbol = action.get("symbol")
        action_id = f"ca_{datetime.now().timestamp()}"
        
        result = {
            "action_id": action_id,
            "type": action_type,
            "symbol": symbol,
            "success": True,
            "details": {}
        }
        
        if action_type == "split":
            ratio = action.get("ratio", 2)
            # This just logs the action - actual position adjustment handled by position manager
            result["details"] = {
                "ratio": ratio,
                "description": f"{ratio}:1 stock split"
            }
            logging.info(f"🔄 Stock split recorded: {symbol} {ratio}:1")
            
        elif action_type == "reverse_split":
            ratio = action.get("ratio", 0.5)  # e.g., 1:2 reverse split
            result["details"] = {
                "ratio": ratio,
                "description": f"{ratio}:1 reverse stock split"
            }
            logging.info(f"🔄 Reverse split recorded: {symbol} {ratio}:1")
            
        elif action_type == "merger":
            new_symbol = action.get("new_symbol")
            ratio = action.get("ratio", 1)
            result["details"] = {
                "new_symbol": new_symbol,
                "ratio": ratio,
                "description": f"Merger into {new_symbol} at {ratio}:1"
            }
            logging.info(f"🔄 Merger recorded: {symbol} -> {new_symbol}")
            
        elif action_type == "spin_off":
            new_symbol = action.get("new_symbol")
            shares_per_share = action.get("shares_per_share", 0.1)
            result["details"] = {
                "new_symbol": new_symbol,
                "shares_per_share": shares_per_share,
                "description": f"Spin-off of {new_symbol}"
            }
            logging.info(f"🔄 Spin-off recorded: {symbol} spinning off {new_symbol}")
            
        elif action_type == "name_change":
            new_symbol = action.get("new_symbol")
            result["details"] = {
                "new_symbol": new_symbol,
                "description": f"Name change to {new_symbol}"
            }
            logging.info(f"🔄 Name change recorded: {symbol} -> {new_symbol}")
        
        # Record corporate action
        action_record = {
            "action_id": action_id,
            "timestamp": datetime.now().isoformat(),
            **action,
            "result": result
        }
        self.corporate_actions.append(action_record)
        
        # Record transaction
        self._add_transaction({
            "timestamp": datetime.now().isoformat(),
            "type": "corporate_action",
            "action_id": action_id,
            "action_type": action_type,
            "symbol": symbol,
            "details": action
        })
        
        return result
    
    def settle_pending(self) -> List[Dict[str, Any]]:
        """
        Process pending settlements that are due
        """
        now = datetime.now()
        settled = []
        
        remaining = []
        for settlement in self.pending_settlements:
            settle_date = datetime.fromisoformat(settlement["settlement_date"])
            
            if now >= settle_date:
                # Settlement complete
                settlement["status"] = "settled"
                settlement["settled_at"] = now.isoformat()
                settled.append(settlement)
                
                # For margin accounts, adjust cash at settlement
                if self.margin_enabled:
                    if settlement["type"] == "debit":
                        self.cash_balance -= settlement["amount"]
                    else:
                        self.cash_balance += settlement["amount"]
                
            else:
                remaining.append(settlement)
        
        self.pending_settlements = remaining
        
        if settled:
            total_settled = sum(s["amount"] for s in settled)
            logging.info(f"💰 Settled {len(settled)} transactions worth ${total_settled:,.2f}")
        
        return settled
    
    def get_available_cash(self) -> float:
        """
        Get cash available for trading
        """
        if self.margin_enabled:
            return self.cash_balance * self.margin_multiplier
        else:
            return self.cash_balance
    
    def get_buying_power(self, positions_value: float = 0) -> float:
        """
        Calculate buying power
        """
        if self.margin_enabled:
            # Simple Reg T margin (50% initial)
            return self.cash_balance + (positions_value * 0.5)
        else:
            return self.cash_balance
    
    def get_cash_flow(self, days: int = 30) -> Dict[str, Any]:
        """
        Get cash flow analysis for period
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        inflows = []
        outflows = []
        total_inflow = 0
        total_outflow = 0
        
        for trans in self.transactions:
            trans_time = datetime.fromisoformat(trans["timestamp"])
            if trans_time >= cutoff:
                if trans.get("cash_impact", 0) > 0:
                    inflows.append(trans)
                    total_inflow += trans["cash_impact"]
                elif trans.get("cash_impact", 0) < 0:
                    outflows.append(trans)
                    total_outflow += abs(trans["cash_impact"])
        
        return {
            "period_days": days,
            "total_inflow": total_inflow,
            "total_outflow": total_outflow,
            "net_cash_flow": total_inflow - total_outflow,
            "inflow_count": len(inflows),
            "outflow_count": len(outflows),
            "average_inflow": total_inflow / len(inflows) if inflows else 0,
            "average_outflow": total_outflow / len(outflows) if outflows else 0,
            "largest_inflow": max([t["cash_impact"] for t in inflows]) if inflows else 0,
            "largest_outflow": max([abs(t["cash_impact"]) for t in outflows]) if outflows else 0
        }
    
    def get_settlement_schedule(self) -> Dict[str, Any]:
        """
        Get upcoming settlement schedule
        """
        now = datetime.now()
        
        # Group by date
        by_date = {}
        total_pending = 0
        total_debits = 0
        total_credits = 0
        
        for settlement in self.pending_settlements:
            settle_date = settlement["settlement_date"][:10]  # YYYY-MM-DD
            if settle_date not in by_date:
                by_date[settle_date] = {
                    "debits": [],
                    "credits": [],
                    "total_debits": 0,
                    "total_credits": 0,
                    "net": 0
                }
            
            if settlement["type"] == "debit":
                by_date[settle_date]["debits"].append(settlement)
                by_date[settle_date]["total_debits"] += settlement["amount"]
                total_debits += settlement["amount"]
            else:
                by_date[settle_date]["credits"].append(settlement)
                by_date[settle_date]["total_credits"] += settlement["amount"]
                total_credits += settlement["amount"]
            
            by_date[settle_date]["net"] = (
                by_date[settle_date]["total_credits"] - 
                by_date[settle_date]["total_debits"]
            )
            total_pending += 1
        
        # Sort by date
        sorted_dates = sorted(by_date.keys())
        
        return {
            "total_pending": total_pending,
            "total_debits": total_debits,
            "total_credits": total_credits,
            "net_settlement": total_credits - total_debits,
            "by_date": {date: by_date[date] for date in sorted_dates},
            "next_settlement": sorted_dates[0] if sorted_dates else None,
            "last_settlement_date": max([s["settlement_date"] for s in self.pending_settlements]) if self.pending_settlements else None
        }
    
    def get_cash_summary(self) -> Dict[str, Any]:
        """
        Get cash summary
        """
        pending = self.get_settlement_schedule()
        
        return {
            "as_of": datetime.now().isoformat(),
            "cash_balance": self.cash_balance,
            "initial_cash": self.initial_cash,
            "total_pnl": self.cash_balance - self.initial_cash,
            "total_pnl_percent": ((self.cash_balance - self.initial_cash) / self.initial_cash) * 100,
            "pending_settlements": pending["total_pending"],
            "pending_net": pending["net_settlement"],
            "projected_cash": self.cash_balance + pending["net_settlement"],
            "available_cash": self.get_available_cash(),
            "buying_power": self.get_buying_power(),
            "margin_enabled": self.margin_enabled
        }
    
    def get_transaction_history(self, limit: int = 100, 
                               transaction_type: str = None) -> List[Dict[str, Any]]:
        """
        Get transaction history
        """
        transactions = self.transactions
        
        if transaction_type:
            transactions = [t for t in transactions if t.get("type") == transaction_type]
        
        return sorted(transactions, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def _add_transaction(self, transaction: Dict[str, Any]):
        """Add a transaction to history"""
        self.transactions.append(transaction)
        
        # Trim if needed
        if len(self.transactions) > self.max_transactions:
            self.transactions = self.transactions[-self.max_transactions:]