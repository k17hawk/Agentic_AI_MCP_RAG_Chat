"""
Options Flow Client - Detects unusual options activity
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import yfinance as yf                     # Added for fallback
import math

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

class OptionsFlowClient:
    """
    Detects and analyzes unusual options activity

    Sources:
    - Unusual options volume
    - Large block trades
    - Sweep orders
    - Put/Call ratios
    - Open interest changes
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # API keys
        self.market_data_api = config.get("market_data_api")
        self.fmp_key = config.get("fmp_key")

        # Rate limiting
        self.rate_limit = config.get("rate_limit", 30)
        self.request_timestamps = []

        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 15) * 60

        # Thresholds
        self.volume_threshold = config.get("volume_threshold", 2.0)  # 2x average
        self.premium_threshold = config.get("premium_threshold", 100000)  # $100k

        logging.info(f"✅ OptionsFlowClient initialized")

    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for unusual options activity
        """
        options = options or {}

        # Extract ticker from query
        import re
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
        ticker = ticker_match.group() if ticker_match else query.upper().replace(" ", "")

        logging.info(f"📊 Options flow search for: '{ticker}'")

        # Check cache
        cache_key = f"options_{ticker}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return cached_result

        await self._rate_limit()

        all_flows = []

        # Try multiple sources
        tasks = []
        if self.fmp_key:
            tasks.append(self._fetch_from_fmp(ticker, options))

        if not tasks:
            return {
                "items": [],
                "all_flows_count": 0,
                "unusual_count": 0,
                "metrics": {},
                "metadata": {
                    "ticker": ticker,
                    "period_hours": options.get("hours_back", 24),
                    "error": "No data sources configured"
                }
            }

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_flows.extend(result)
            elif isinstance(result, Exception):
                logging.warning(f"⚠️ Options flow error: {result}")

        # Filter unusual activity
        unusual_flows = self._filter_unusual(all_flows)

        # Sort by significance
        unusual_flows.sort(key=lambda x: x.get("significance_score", 0), reverse=True)

        # Calculate aggregate metrics
        metrics = self._calculate_metrics(all_flows, unusual_flows)

        result = {
            "items": unusual_flows[:options.get("max_results", 50)],
            "all_flows_count": len(all_flows),
            "unusual_count": len(unusual_flows),
            "metrics": metrics,
            "metadata": {
                "ticker": ticker,
                "period_hours": options.get("hours_back", 24)
            }
        }

        self.cache[cache_key] = (datetime.now().timestamp(), result)

        logging.info(f"✅ Options flow: {len(unusual_flows)} unusual trades detected for {ticker}")
        return result

    # ----------------------------------------------------------------------
    # FIXED: Correct FMP endpoint and parsing
    # ----------------------------------------------------------------------
    async def _fetch_from_fmp(self, ticker: str, options: Dict) -> List[Dict]:
        """Fetch options data from FMP (correct endpoint) with yfinance fallback."""
        flows = []

        # --- Attempt FMP (correct endpoint) ---
        if self.fmp_key:
            # Correct endpoint: /v3/option-chain/{symbol}
            url = f"https://financialmodelingprep.com/api/v3/option-chain/{ticker}"
            params = {"apikey": self.fmp_key}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            # New endpoint returns a list of options
                            if isinstance(data, list) and data:
                                for option in data:
                                    # Determine option type from contractType field
                                    opt_type = option.get("contractType", "").lower()
                                    if opt_type in ("call", "put"):
                                        # Pass ticker explicitly (the option dict may not have it)
                                        flow = self._analyze_option(option, opt_type, ticker)
                                        if flow:
                                            flows.append(flow)
                                logging.info(f"✅ FMP options: {len(flows)} items for {ticker}")
                                return flows   # success, return early
                            else:
                                logging.debug(f"FMP returned no data or unexpected format for {ticker}")
            except asyncio.TimeoutError:
                logging.debug(f"FMP options timeout for {ticker}")
            except Exception as e:
                logging.debug(f"FMP options error (fallback triggered): {e}")

        # --- Fallback to yfinance if FMP failed or returned no data ---
        logging.info(f"📡 Falling back to yfinance for options data on {ticker}")
        yf_flows = await self._fetch_options_from_yfinance(ticker)
        if yf_flows:
            flows.extend(yf_flows)
            logging.info(f"✅ yfinance options: {len(yf_flows)} items for {ticker}")

        return flows

    async def _fetch_options_from_yfinance(self, ticker: str) -> List[Dict]:
        """Fetch options data using yfinance (runs in thread pool) and analyze."""
        loop = asyncio.get_event_loop()

        def _get_options_blocking():
            try:
                stock = yf.Ticker(ticker)
                expirations = stock.options
                if not expirations:
                    return []

                # Use nearest expiration
                chain = stock.option_chain(expirations[0])
                flows = []

                # Process calls
                for opt in chain.calls.to_dict('records'):
                    # Add required fields that yfinance might omit
                    opt["symbol"] = ticker
                    opt["expirationDate"] = expirations[0]
                    analyzed = self._analyze_option(opt, "call", ticker)
                    if analyzed:
                        analyzed["source"] = "yfinance"   # override source
                        flows.append(analyzed)

                # Process puts
                for opt in chain.puts.to_dict('records'):
                    opt["symbol"] = ticker
                    opt["expirationDate"] = expirations[0]
                    analyzed = self._analyze_option(opt, "put", ticker)
                    if analyzed:
                        analyzed["source"] = "yfinance"
                        flows.append(analyzed)

                return flows
            except Exception as e:
                logging.debug(f"yfinance options error for {ticker}: {e}")
                return []

        try:
            flows = await asyncio.wait_for(
                loop.run_in_executor(None, _get_options_blocking),
                timeout=10.0
            )
            return flows
        except asyncio.TimeoutError:
            logging.warning(f"⏰ yfinance options timeout for {ticker}")
            return []
        except Exception as e:
            logging.warning(f"⚠️ yfinance options exception: {e}")
            return []

    # ----------------------------------------------------------------------
    # FIXED: Safer number conversion and explicit ticker parameter
    # ----------------------------------------------------------------------
    def _analyze_option(self, option: Dict, option_type: str, ticker: str = "") -> Optional[Dict]:
        """
        Analyze a single option for unusual activity.
        Handles NaN/None values gracefully and adds title/content for display.
        """
        try:
            # Safely extract values using helper methods
            strike = self._safe_float(option.get("strike"))
            expiration = option.get("expirationDate", "")

            volume = self._safe_int(option.get("volume"))
            open_interest = self._safe_int(option.get("openInterest"))
            last_price = self._safe_float(option.get("lastPrice"))
            implied_volatility = self._safe_float(option.get("impliedVolatility"))

            # Use provided ticker or fallback to option dict
            ticker = ticker or option.get("symbol", "")

            # Calculate premium
            premium = volume * last_price * 100  # Each contract is 100 shares

            # Calculate volume ratio (rough estimate)
            avg_volume = max(open_interest * 0.1, 1)  # avoid division by zero
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0

            # Determine if unusual
            is_unusual = (
                volume_ratio > self.volume_threshold or
                premium > self.premium_threshold
            )

            # Calculate significance score (0-100)
            significance = 0.0
            if is_unusual:
                significance = min(100.0, (
                    volume_ratio * 20 +
                    (premium / 100000) * 30 +
                    (implied_volatility * 20)
                ))

            # Build human-readable title and content
            action = "Unusual" if is_unusual else "Normal"
            opt_type_display = "CALL" if option_type == "call" else "PUT"
            title = f"{action} {opt_type_display} option on {ticker} at ${strike:.2f}"
            content = (
                f"Option Type: {opt_type_display}\n"
                f"Strike: ${strike:.2f}\n"
                f"Expiration: {expiration}\n"
                f"Volume: {volume:,}\n"
                f"Open Interest: {open_interest:,}\n"
                f"Last Price: ${last_price:.2f}\n"
                f"Premium: ${premium:,.2f}\n"
                f"Volume Ratio: {volume_ratio:.2f}x\n"
                f"Implied Volatility: {implied_volatility:.2%}\n"
                f"Significance Score: {significance:.1f}"
            )

            return {
                "ticker": ticker,
                "option_type": option_type,
                "strike": strike,
                "expiration": expiration,
                "volume": volume,
                "open_interest": open_interest,
                "last_price": last_price,
                "premium": premium,
                "volume_ratio": volume_ratio,
                "implied_volatility": implied_volatility,
                "is_unusual": is_unusual,
                "significance_score": significance,
                "source": "fmp",  # will be overridden in yfinance branch
                "timestamp": datetime.now().isoformat(),
                # New fields for display/enrichment
                "title": title,
                "content": content,
                "relevance_score": 1.0 if is_unusual else 0.5  # default, enricher may override
            }
        except Exception as e:
            logging.debug(f"Option analysis error: {e}")
            return None

    # ----------------------------------------------------------------------
    # NEW: Safe number conversion helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _safe_float(value, default=0.0):
        """Convert value to float, return default on failure."""
        if value is None:
            return default
        try:
            f = float(value)
            return f if not math.isnan(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value, default=0):
        """Convert value to int, return default on failure."""
        if value is None:
            return default
        try:
            # Handle float strings like "100.0"
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return int(float(value))  # in case value is "100.0"
        except (ValueError, TypeError):
            return default

    # ----------------------------------------------------------------------
    # Legacy helper kept for compatibility (not used internally now)
    # ----------------------------------------------------------------------
    @staticmethod
    def _is_nan(value):
        """Legacy method – prefer _safe_float / _safe_int."""
        try:
            return math.isnan(float(value))
        except:
            return False

    def _filter_unusual(self, flows: List[Dict]) -> List[Dict]:
        """Filter for unusual options activity"""
        unusual = []

        for flow in flows:
            if flow.get("is_unusual"):
                unusual.append(flow)

        return unusual

    def _calculate_metrics(self, all_flows: List[Dict], unusual_flows: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate options metrics"""
        if not all_flows:
            return {}

        # Put/Call ratio
        puts = sum(1 for f in all_flows if f.get("option_type") == "put")
        calls = sum(1 for f in all_flows if f.get("option_type") == "call")
        put_call_ratio = puts / calls if calls > 0 else float('inf')

        # Unusual put/call ratio
        unusual_puts = sum(1 for f in unusual_flows if f.get("option_type") == "put")
        unusual_calls = sum(1 for f in unusual_flows if f.get("option_type") == "call")
        unusual_pcr = unusual_puts / unusual_calls if unusual_calls > 0 else float('inf')

        # Total premiums
        total_premium = sum(f.get("premium", 0) for f in all_flows)
        unusual_premium = sum(f.get("premium", 0) for f in unusual_flows)

        return {
            "total_volume": len(all_flows),
            "puts": puts,
            "calls": calls,
            "put_call_ratio": float(put_call_ratio),
            "unusual_puts": unusual_puts,
            "unusual_calls": unusual_calls,
            "unusual_put_call_ratio": float(unusual_pcr) if unusual_calls > 0 else None,
            "total_premium": float(total_premium),
            "unusual_premium": float(unusual_premium),
            "unusual_percentage": len(unusual_flows) / len(all_flows) * 100 if all_flows else 0
        }

    async def _rate_limit(self):
        """Rate limiting"""
        now = datetime.now().timestamp()

        self.request_timestamps = [ts for ts in self.request_timestamps
                                  if now - ts < 60]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)