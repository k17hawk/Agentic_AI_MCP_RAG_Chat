"""
Macro Data Client - Fetches macroeconomic indicators
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import pandas as pd
import numpy as np

from utils.logger import logger as logging
from utils.decorators import retry

class MacroDataClient:
    """
    Fetches macroeconomic indicators from multiple sources
    
    Data types:
    - GDP, Inflation, Unemployment
    - Interest rates (Fed, ECB, etc.)
    - Consumer sentiment
    - PMI (Manufacturing, Services)
    - Housing data
    - Trade data
    - Central bank policies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys
        self.fred_api_key = config.get("fred_api_key")  # St. Louis Fed
        self.worldbank_api_key = config.get("worldbank_api_key")
        self.imf_api_key = config.get("imf_api_key")
        self.oecd_api_key = config.get("oecd_api_key")
        self.econdb_api_key = config.get("econdb_api_key")
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 20)
        self.request_timestamps = []
        
        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_hours", 24) * 3600
        
        # Common indicators by category
        self.indicators = config.get("indicators", {
            # Economic Growth
            "gdp": {
                "fred": "GDP",
                "name": "Gross Domestic Product",
                "unit": "billions",
                "frequency": "quarterly"
            },
            "gdp_growth": {
                "fred": "A191RL1Q225SBEA",
                "name": "Real GDP Growth",
                "unit": "percent",
                "frequency": "quarterly"
            },
            
            # Inflation
            "cpi": {
                "fred": "CPIAUCSL",
                "name": "Consumer Price Index",
                "unit": "index",
                "frequency": "monthly"
            },
            "core_cpi": {
                "fred": "CPILFESL",
                "name": "Core CPI (ex-food & energy)",
                "unit": "index",
                "frequency": "monthly"
            },
            "ppi": {
                "fred": "PPIACO",
                "name": "Producer Price Index",
                "unit": "index",
                "frequency": "monthly"
            },
            "inflation_rate": {
                "fred": "FPCPITOTLZGUSA",
                "name": "Inflation Rate",
                "unit": "percent",
                "frequency": "annual"
            },
            
            # Labor Market
            "unemployment": {
                "fred": "UNRATE",
                "name": "Unemployment Rate",
                "unit": "percent",
                "frequency": "monthly"
            },
            "nonfarm_payrolls": {
                "fred": "PAYEMS",
                "name": "Nonfarm Payrolls",
                "unit": "thousands",
                "frequency": "monthly"
            },
            "labor_force_participation": {
                "fred": "CIVPART",
                "name": "Labor Force Participation Rate",
                "unit": "percent",
                "frequency": "monthly"
            },
            "job_openings": {
                "fred": "JTSJOL",
                "name": "Job Openings",
                "unit": "thousands",
                "frequency": "monthly"
            },
            
            # Interest Rates
            "fed_funds": {
                "fred": "FEDFUNDS",
                "name": "Federal Funds Rate",
                "unit": "percent",
                "frequency": "daily"
            },
            "treasury_2y": {
                "fred": "GS2",
                "name": "2-Year Treasury Yield",
                "unit": "percent",
                "frequency": "daily"
            },
            "treasury_10y": {
                "fred": "GS10",
                "name": "10-Year Treasury Yield",
                "unit": "percent",
                "frequency": "daily"
            },
            "treasury_30y": {
                "fred": "GS30",
                "name": "30-Year Treasury Yield",
                "unit": "percent",
                "frequency": "daily"
            },
            "yield_curve": {
                "fred": "T10Y2Y",
                "name": "10Y-2Y Yield Curve",
                "unit": "percent",
                "frequency": "daily"
            },
            
            # Consumer
            "consumer_sentiment": {
                "fred": "UMCSENT",
                "name": "Consumer Sentiment",
                "unit": "index",
                "frequency": "monthly"
            },
            "retail_sales": {
                "fred": "RSAFS",
                "name": "Retail Sales",
                "unit": "millions",
                "frequency": "monthly"
            },
            "personal_income": {
                "fred": "PI",
                "name": "Personal Income",
                "unit": "billions",
                "frequency": "monthly"
            },
            "personal_saving_rate": {
                "fred": "PSAVERT",
                "name": "Personal Saving Rate",
                "unit": "percent",
                "frequency": "monthly"
            },
            
            # Housing
            "housing_starts": {
                "fred": "HOUST",
                "name": "Housing Starts",
                "unit": "thousands",
                "frequency": "monthly"
            },
            "building_permits": {
                "fred": "PERMIT",
                "name": "Building Permits",
                "unit": "thousands",
                "frequency": "monthly"
            },
            "existing_home_sales": {
                "fred": "EXHOSLUSM495S",
                "name": "Existing Home Sales",
                "unit": "units",
                "frequency": "monthly"
            },
            "new_home_sales": {
                "fred": "HSN1F",
                "name": "New Home Sales",
                "unit": "thousands",
                "frequency": "monthly"
            },
            "home_prices": {
                "fred": "CSUSHPINSA",
                "name": "S&P/Case-Shiller Home Price Index",
                "unit": "index",
                "frequency": "monthly"
            },
            
            # Manufacturing
            "manufacturing_pmi": {
                "fred": "NAPM",
                "name": "ISM Manufacturing PMI",
                "unit": "index",
                "frequency": "monthly"
            },
            "industrial_production": {
                "fred": "INDPRO",
                "name": "Industrial Production",
                "unit": "index",
                "frequency": "monthly"
            },
            "capacity_utilization": {
                "fred": "TCU",
                "name": "Capacity Utilization",
                "unit": "percent",
                "frequency": "monthly"
            },
            "factory_orders": {
                "fred": "AMTMNO",
                "name": "Factory Orders",
                "unit": "millions",
                "frequency": "monthly"
            },
            "durable_goods": {
                "fred": "DGORDER",
                "name": "Durable Goods Orders",
                "unit": "millions",
                "frequency": "monthly"
            },
            
            # Trade
            "trade_deficit": {
                "fred": "BOPGSTB",
                "name": "Trade Balance",
                "unit": "millions",
                "frequency": "monthly"
            },
            "exports": {
                "fred": "EXPGS",
                "name": "Exports",
                "unit": "millions",
                "frequency": "monthly"
            },
            "imports": {
                "fred": "IMPGS",
                "name": "Imports",
                "unit": "millions",
                "frequency": "monthly"
            },
            
            # Confidence
            "business_confidence": {
                "oecd": "BSRE",
                "name": "Business Confidence",
                "unit": "index",
                "frequency": "monthly"
            },
            "consumer_confidence": {
            "oecd": "CSCICP03",
                "name": "Consumer Confidence",
                "unit": "index",
                "frequency": "monthly"
            },
            "economic_sentiment": {
                "oecd": "ESD",
                "name": "Economic Sentiment",
                "unit": "index",
                "frequency": "monthly"
            }
        })
        
        # Country codes
        self.country_codes = config.get("country_codes", {
            "us": "USA",
            "usa": "USA",
            "united states": "USA",
            "china": "CHN",
            "japan": "JPN",
            "germany": "DEU",
            "uk": "GBR",
            "france": "FRA",
            "italy": "ITA",
            "canada": "CAN",
            "australia": "AUS"
        })
        
        logging.info(f"✅ MacroDataClient initialized with {len(self.indicators)} indicators")
    
    @retry(max_attempts=3, delay=2.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for macroeconomic data
        """
        options = options or {}
        logging.info(f"📈 Macro data search for: '{query}'")
        
        # Check cache
        cache_key = f"macro_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached macro data for '{query}'")
                return cached_result
        
        await self._rate_limit()
        
        all_data = []
        sources_used = []
        
        # Determine country and indicator
        country = options.get("country", "us").lower()
        country_code = self.country_codes.get(country, "USA")
        
        indicator_name = options.get("indicator", query.lower())
        
        # Try FRED first (US data)
        if self.fred_api_key:
            fred_data = await self._fetch_from_fred(indicator_name, country_code, options)
            if fred_data:
                all_data.append(fred_data)
                sources_used.append("fred")
        
        # Try OECD
        if self.oecd_api_key:
            oecd_data = await self._fetch_from_oecd(indicator_name, country_code, options)
            if oecd_data:
                all_data.append(oecd_data)
                sources_used.append("oecd")
        
        # Try World Bank
        if self.worldbank_api_key:
            wb_data = await self._fetch_from_worldbank(indicator_name, country_code, options)
            if wb_data:
                all_data.append(wb_data)
                sources_used.append("worldbank")
        
        # Try IMF
        if self.imf_api_key:
            imf_data = await self._fetch_from_imf(indicator_name, country_code, options)
            if imf_data:
                all_data.append(imf_data)
                sources_used.append("imf")
        
        # Calculate composite metrics
        composite = self._calculate_composite_metrics(all_data)
        
        result = {
            "query": query,
            "country": country,
            "country_code": country_code,
            "timestamp": datetime.now().isoformat(),
            "items": all_data,
            "sources_used": sources_used,
            "composite": composite,
            "metadata": {
                "total_sources": len(all_data),
                "data_points": sum(len(d.get("observations", [])) for d in all_data)
            }
        }
        
        # Cache result
        self.cache[cache_key] = (datetime.now().timestamp(), result)
        
        logging.info(f"✅ Macro data found from {len(all_data)} sources")
        return result
    
    async def _fetch_from_fred(self, indicator: str, country: str, options: Dict) -> Optional[Dict]:
        """
        Fetch data from FRED (Federal Reserve Economic Data)
        """
        if not self.fred_api_key or country != "USA":
            return None
        
        # Map common names to FRED series IDs
        series_id = self._map_to_fred_id(indicator)
        if not series_id:
            return None
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Calculate date range
        end_date = options.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        
        if "years" in options:
            start_date = (datetime.now() - timedelta(days=options["years"]*365)).strftime("%Y-%m-%d")
        else:
            start_date = options.get("start_date", "1900-01-01")
        
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "limit": options.get("limit", 100000),
            "sort_order": "desc"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        observations = []
                        for obs in data.get("observations", []):
                            if obs.get("value") != ".":
                                observations.append({
                                    "date": obs.get("date"),
                                    "value": float(obs.get("value")),
                                    "realtime_start": obs.get("realtime_start"),
                                    "realtime_end": obs.get("realtime_end")
                                })
                        
                        # Get series info
                        info = await self._get_fred_series_info(series_id)
                        
                        # Calculate statistics
                        values = [o["value"] for o in observations]
                        
                        return {
                            "source": "fred",
                            "series_id": series_id,
                            "indicator": info.get("title", indicator),
                            "units": info.get("units", "units"),
                            "frequency": info.get("frequency", "unknown"),
                            "seasonal_adjustment": info.get("seasonal_adjustment", "NSA"),
                            "observations": observations[:options.get("max_obs", 1000)],
                            "latest_value": observations[0]["value"] if observations else None,
                            "latest_date": observations[0]["date"] if observations else None,
                            "statistics": {
                                "mean": float(np.mean(values)) if values else None,
                                "median": float(np.median(values)) if values else None,
                                "min": float(min(values)) if values else None,
                                "max": float(max(values)) if values else None,
                                "std": float(np.std(values)) if values else None,
                                "trend": self._calculate_trend(values) if values else None
                            }
                        }
        except Exception as e:
            logging.debug(f"FRED error: {e}")
        
        return None
    
    async def _get_fred_series_info(self, series_id: str) -> Dict:
        """Get metadata for FRED series"""
        url = "https://api.stlouisfed.org/fred/series"
        
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        series = data.get("seriess", [{}])[0]
                        return {
                            "title": series.get("title"),
                            "units": series.get("units"),
                            "frequency": series.get("frequency"),
                            "seasonal_adjustment": series.get("seasonal_adjustment"),
                            "notes": series.get("notes")
                        }
        except:
            pass
        
        return {}
    
    def _map_to_fred_id(self, indicator: str) -> Optional[str]:
        """Map common indicator names to FRED series IDs"""
        # Direct match in indicators dict
        for key, value in self.indicators.items():
            if key == indicator.lower() or value.get("name", "").lower() == indicator.lower():
                if "fred" in value:
                    return value["fred"]
        
        # Common mappings
        mapping = {
            "gdp": "GDP",
            "real gdp": "GDPC1",
            "inflation": "CPIAUCSL",
            "cpi": "CPIAUCSL",
            "core cpi": "CPILFESL",
            "unemployment": "UNRATE",
            "fed funds": "FEDFUNDS",
            "interest rate": "FEDFUNDS",
            "consumer sentiment": "UMCSENT",
            "consumer confidence": "UMCSENT",
            "pmi": "NAPM",
            "manufacturing pmi": "NAPM",
            "housing starts": "HOUST",
            "building permits": "PERMIT",
            "retail sales": "RSAFS",
            "industrial production": "INDPRO",
            "capacity utilization": "TCU",
            "trade balance": "BOPGSTB",
            "10 year treasury": "GS10",
            "2 year treasury": "GS2",
            "yield curve": "T10Y2Y"
        }
        
        return mapping.get(indicator.lower())
    
    async def _fetch_from_oecd(self, indicator: str, country: str, options: Dict) -> Optional[Dict]:
        """Fetch data from OECD API"""
        if not self.oecd_api_key:
            return None
        
        # OECD API implementation would go here
        # This is a placeholder
        return None
    
    async def _fetch_from_worldbank(self, indicator: str, country: str, options: Dict) -> Optional[Dict]:
        """Fetch data from World Bank API"""
        if not self.worldbank_api_key:
            return None
        
        # World Bank API implementation
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        
        params = {
            "format": "json",
            "per_page": options.get("limit", 100),
            "date": f"{options.get('start_year', 1960)}:{options.get('end_year', datetime.now().year)}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if len(data) > 1:
                            observations = []
                            for item in data[1]:
                                if item.get("value") is not None:
                                    observations.append({
                                        "date": str(item.get("date")),
                                        "value": float(item.get("value"))
                                    })
                            
                            return {
                                "source": "worldbank",
                                "indicator": data[1][0].get("indicator", {}).get("value", indicator) if data[1] else indicator,
                                "country": country,
                                "observations": observations,
                                "latest_value": observations[0]["value"] if observations else None,
                                "latest_date": observations[0]["date"] if observations else None
                            }
        except Exception as e:
            logging.debug(f"World Bank error: {e}")
        
        return None
    
    async def _fetch_from_imf(self, indicator: str, country: str, options: Dict) -> Optional[Dict]:
        """Fetch data from IMF API"""
        if not self.imf_api_key:
            return None
        
        # IMF API implementation would go here
        # This is a placeholder
        return None
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate trend statistics
        """
        if len(values) < 2:
            return {"direction": "unknown", "strength": 0}
        
        # Recent trend (last 3 vs previous 3)
        recent = values[:3] if len(values) >= 3 else values
        previous = values[3:6] if len(values) >= 6 else values[3:]
        
        if recent and previous:
            recent_avg = sum(recent) / len(recent)
            previous_avg = sum(previous) / len(previous)
            
            change = recent_avg - previous_avg
            change_pct = (change / abs(previous_avg)) * 100 if previous_avg != 0 else 0
            
            direction = "up" if change > 0 else "down" if change < 0 else "flat"
            strength = min(100, abs(change_pct))
            
            return {
                "direction": direction,
                "strength": float(strength),
                "change": float(change),
                "change_percent": float(change_pct)
            }
        
        return {"direction": "unknown", "strength": 0}
    
    def _calculate_composite_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate composite macroeconomic metrics
        """
        composite = {
            "growth_indicators": [],
            "inflation_indicators": [],
            "labor_indicators": [],
            "confidence_indicators": []
        }
        
        for item in data:
            indicator = item.get("indicator", "").lower()
            latest = item.get("latest_value")
            
            if not latest:
                continue
            
            # Categorize indicators
            if any(word in indicator for word in ["gdp", "growth", "production"]):
                composite["growth_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date")
                })
            elif any(word in indicator for word in ["cpi", "inflation", "price"]):
                composite["inflation_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date")
                })
            elif any(word in indicator for word in ["unemployment", "payroll", "labor", "jobs"]):
                composite["labor_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date")
                })
            elif any(word in indicator for word in ["sentiment", "confidence"]):
                composite["confidence_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date")
                })
        
        return composite
    
    async def get_latest_value(self, indicator: str, country: str = "us") -> Optional[float]:
        """
        Get the latest value for an indicator
        """
        result = await self.search(indicator, {
            "country": country,
            "max_obs": 1
        })
        
        if result.get("items"):
            for item in result["items"]:
                if item.get("latest_value") is not None:
                    return item["latest_value"]
        
        return None
    
    async def get_time_series(self, indicator: str, country: str = "us", 
                             years: int = 10) -> List[Dict]:
        """
        Get time series data for an indicator
        """
        result = await self.search(indicator, {
            "country": country,
            "years": years,
            "max_obs": years * 12
        })
        
        for item in result.get("items", []):
            if "observations" in item:
                return item["observations"]
        
        return []
    
    async def compare_countries(self, indicator: str, countries: List[str]) -> Dict[str, Any]:
        """
        Compare an indicator across multiple countries
        """
        results = {}
        
        for country in countries:
            value = await self.get_latest_value(indicator, country)
            if value is not None:
                results[country] = value
        
        return {
            "indicator": indicator,
            "timestamp": datetime.now().isoformat(),
            "values": results,
            "ranked": sorted(results.items(), key=lambda x: x[1], reverse=True)
        }
    
    async def get_economic_calendar(self, days: int = 30) -> List[Dict]:
        """
        Get upcoming economic releases
        """
        # This would fetch from an economic calendar API
        # Placeholder implementation
        return []
    
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