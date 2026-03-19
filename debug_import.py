"""
Drop this file in your project root and run:
  python debug_imports.py
It will print the EXACT file paths Python is importing for each module.
"""
import sys, importlib

# Add project root to path (same as editable install does)
modules_to_check = [
    "agentic_trading_system.discovery.entity_extractor.nlp_extractor",
    "agentic_trading_system.discovery.entity_extractor.regex_extractor",
    "agentic_trading_system.discovery.data_enricher",
]

print("=== sys.path ===")
for p in sys.path:
    print(f"  {p}")

print("\n=== Module file locations ===")
for mod_name in modules_to_check:
    try:
        mod = importlib.import_module(mod_name)
        print(f"  {mod_name}")
        print(f"    -> {mod.__file__}")
        # Check if fix is present
        if "nlp_extractor" in mod_name:
            has_fix = hasattr(mod.NLPExtractor, '_FRAGMENT_WORDS')
            has_new_pattern = hasattr(mod.NLPExtractor({}), '_ticker_pattern')
            print(f"    _FRAGMENT_WORDS present: {has_fix}")
            print(f"    _ticker_pattern present: {has_new_pattern}")
        elif "regex_extractor" in mod_name:
            # Check blacklist has IDEAS/QUOTE/CCPA
            inst = mod.RegexExtractor({})
            has_ideas = 'IDEAS' in inst.ticker_blacklist
            has_ccpa  = 'CCPA'  in inst.ticker_blacklist
            has_stock = 'STOCK' in inst.ticker_blacklist
            print(f"    IDEAS in blacklist: {has_ideas}")
            print(f"    CCPA  in blacklist: {has_ccpa}")
            print(f"    STOCK in blacklist: {has_stock}")
    except Exception as e:
        print(f"  {mod_name}")
        print(f"    ERROR: {e}")

print("\n=== Done ===")