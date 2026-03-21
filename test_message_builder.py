async def test_message_builder(self):
    """
    Test 1: Message Builder
    """
    print(f"\n   📝 Testing Message Builder:")
    print("   " + "-" * 40)
    
    trade = {
        "symbol": "AAPL",
        "action": "BUY",
        "price": 247.99,
        "confidence": 0.75,
        "shares": 6,
        "position_value": 1664,
        "stop_loss": 235.59,
        "take_profit": 260.00,
        "rr_ratio": 2.5,
        "reasons": ["Strong technical signal", "Good fundamentals", "Bullish market trend"],
        "concerns": ["Market volatility", "Sector rotation"]
    }
    
    whatsapp_msg = self.message_builder.build_approval_request(trade)
    
    print(f"      • WhatsApp Message Preview:")
    print(f"         {whatsapp_msg[:200]}...")
    
    return {"whatsapp": whatsapp_msg}