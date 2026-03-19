"""
Macro Data Client - Fetches macroeconomic indicators
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import pandas as pd
import numpy as np

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

class MacroDataClient:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys (only FRED requires a key)
        self.fred_api_key = config.get("fred_api_key") 
        
        # All these sources are free and don't require API keys
        self.worldbank_enabled = config.get("worldbank_enabled", True)
        self.oecd_enabled = config.get("oecd_enabled", True)
        self.imf_enabled = config.get("imf_enabled", True)
        
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
                "worldbank": "NY.GDP.MKTP.CD",
                "oecd": "GDP",
                "imf": "NGDPD",  # IMF code for GDP
                "name": "Gross Domestic Product",
                "unit": "billions",
                "frequency": "quarterly"
            },
            "gdp_growth": {
                "fred": "A191RL1Q225SBEA",
                "worldbank": "NY.GDP.MKTP.KD.ZG",
                "oecd": "GDPV",
                "imf": "NGDP_R",
                "name": "Real GDP Growth",
                "unit": "percent",
                "frequency": "quarterly"
            },
            
            # Inflation
            "cpi": {
                "fred": "CPIAUCSL",
                "worldbank": "FP.CPI.TOTL",
                "oecd": "CPI",
                "imf": "PCPI",
                "name": "Consumer Price Index",
                "unit": "index",
                "frequency": "monthly"
            },
            "core_cpi": {
                "fred": "CPILFESL",
                "imf": "PCPIP",
                "name": "Core CPI (ex-food & energy)",
                "unit": "index",
                "frequency": "monthly"
            },
            "ppi": {
                "fred": "PPIACO",
                "imf": "PPI",
                "name": "Producer Price Index",
                "unit": "index",
                "frequency": "monthly"
            },
            "inflation_rate": {
                "fred": "FPCPITOTLZGUSA",
                "worldbank": "FP.CPI.TOTL.ZG",
                "oecd": "CPALTT01",
                "imf": "PCPIPCH",
                "name": "Inflation Rate",
                "unit": "percent",
                "frequency": "annual"
            },
            
            # Labor Market
            "unemployment": {
                "fred": "UNRATE",
                "worldbank": "SL.UEM.TOTL.ZS",
                "oecd": "LRHUTTTT",
                "imf": "LUR",
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
                "oecd": "LF",
                "imf": "LFP",
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
                "oecd": "PMI",
                "imf": "PMI",
                "name": "ISM Manufacturing PMI",
                "unit": "index",
                "frequency": "monthly"
            },
            "industrial_production": {
                "fred": "INDPRO",
                "oecd": "PRINTO",
                "imf": "IIP",
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
                "worldbank": "NE.RSB.GNFS.CD",
                "oecd": "B1_GE",
                "imf": "BCA",
                "name": "Trade Balance",
                "unit": "millions",
                "frequency": "monthly"
            },
            "exports": {
                "fred": "EXPGS",
                "worldbank": "NE.EXP.GNFS.CD",
                "imf": "BX",
                "name": "Exports",
                "unit": "millions",
                "frequency": "monthly"
            },
            "imports": {
                "fred": "IMPGS",
                "worldbank": "NE.IMP.GNFS.CD",
                "imf": "BM",
                "name": "Imports",
                "unit": "millions",
                "frequency": "monthly"
            },
            
            # Confidence
            "business_confidence": {
                "oecd": "BSRE",
                "imf": "BCI",
                "name": "Business Confidence",
                "unit": "index",
                "frequency": "monthly"
            },
            "consumer_confidence": {
                "oecd": "CSCICP03",
                "imf": "CCI",
                "name": "Consumer Confidence",
                "unit": "index",
                "frequency": "monthly"
            },
            "economic_sentiment": {
                "oecd": "ESD",
                "imf": "ESI",
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
            "united kingdom": "GBR",
            "france": "FRA",
            "italy": "ITA",
            "canada": "CAN",
            "australia": "AUS",
            "brazil": "BRA",
            "india": "IND",
            "russia": "RUS",
            "korea": "KOR",
            "mexico": "MEX"
        })
        
        # OECD country codes mapping
        self.oecd_country_codes = config.get("oecd_country_codes", {
            "USA": "USA",
            "GBR": "GBR",
            "DEU": "DEU",
            "FRA": "FRA",
            "ITA": "ITA",
            "CAN": "CAN",
            "JPN": "JPN",
            "AUS": "AUS",
            "KOR": "KOR",
            "MEX": "MEX"
        })
        
        # IMF country codes mapping (ISO 3166-1 alpha-3)
        self.imf_country_codes = config.get("imf_country_codes", {
            "USA": "US",
            "GBR": "GB",
            "DEU": "DE",
            "FRA": "FR",
            "ITA": "IT",
            "CAN": "CA",
            "JPN": "JP",
            "AUS": "AU",
            "CHN": "CN",
            "IND": "IN",
            "BRA": "BR",
            "RUS": "RU",
            "KOR": "KR",
            "MEX": "MX"
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
        
        # Try FRED first (US data) - requires API key
        if self.fred_api_key:
            fred_data = await self._fetch_from_fred(indicator_name, country_code, options)
            if fred_data:
                all_data.append(fred_data)
                sources_used.append("fred")
        
        # Try OECD - no API key required
        if self.oecd_enabled:
            oecd_data = await self._fetch_from_oecd(indicator_name, country_code, options)
            if oecd_data:
                all_data.append(oecd_data)
                sources_used.append("oecd")
        
        # Try World Bank - no API key required
        if self.worldbank_enabled:
            wb_data = await self._fetch_from_worldbank(indicator_name, country_code, options)
            if wb_data:
                all_data.append(wb_data)
                sources_used.append("worldbank")
        
        # Try IMF - no API key required (SDMX 2.1 and 3.0 APIs)
        if self.imf_enabled:
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
        Requires API key (free registration)
        """
        if not self.fred_api_key:
            logging.debug("FRED API key not configured")
            return None
        
        if country != "USA":
            logging.debug(f"FRED only supports USA data, requested: {country}")
            return None
        
        # Map common names to FRED series IDs
        series_id = self._map_to_fred_id(indicator)
        if not series_id:
            logging.debug(f"No FRED series ID found for indicator: {indicator}")
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
                        
                        if not observations:
                            return None
                        
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
                    else:
                        logging.debug(f"FRED API error: {response.status}")
                        return None
        except Exception as e:
            logging.debug(f"FRED error: {e}")
        
        return None
    
    async def _get_fred_series_info(self, series_id: str) -> Dict:
        """Get metadata for FRED series"""
        if not self.fred_api_key:
            return {}
        
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
        """
        Fetch data from OECD API
        No API key required - free and open data
        """
        if not self.oecd_enabled:
            return None
        
        # Check if country is supported by OECD
        if country not in self.oecd_country_codes.values():
            logging.debug(f"Country {country} not in OECD coverage")
            return None
        
        # Map indicator to OECD code
        oecd_code = self._map_to_oecd_code(indicator)
        if not oecd_code:
            logging.debug(f"No OECD code found for indicator: {indicator}")
            return None
        
        # OECD API endpoint using SDMX-JSON
        # Using the .Stat API which is more straightforward
        url = "https://stats.oecd.org/SDMX-JSON/data"
        
        # Simplified mapping - in production, you'd need proper dimension mapping
        dataset_map = {
            "CPI": "PRICES_CPI",  # Consumer Price Indices
            "GDP": "SNA_TABLE1",   # Gross Domestic Product
            "UNRATE": "STLABOUR",  # Labour Force Statistics
            "PMI": "MEI_CLI",      # Composite Leading Indicators
            "CLI": "MEI_CLI",      # Composite Leading Indicators
            "BSRE": "MEI_BS"       # Business Surveys
        }
        
        dataset = dataset_map.get(oecd_code, "MEI_CLI")
        
        # Build URL with proper parameters
        full_url = f"{url}/{dataset}/{country}.{oecd_code}.all?startTime={options.get('start_year', 2000)}&endTime={options.get('end_year', datetime.now().year)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        observations = self._parse_oecd_response(data, oecd_code)
                        
                        if not observations:
                            return None
                        
                        values = [obs["value"] for obs in observations]
                        
                        return {
                            "source": "oecd",
                            "indicator": self._get_indicator_name(oecd_code),
                            "country": country,
                            "frequency": options.get("frequency", "monthly"),
                            "observations": observations[:options.get("max_obs", 100)],
                            "latest_value": observations[0]["value"] if observations else None,
                            "latest_date": observations[0]["date"] if observations else None,
                            "statistics": {
                                "mean": float(np.mean(values)) if values else None,
                                "median": float(np.median(values)) if values else None,
                                "min": float(min(values)) if values else None,
                                "max": float(max(values)) if values else None,
                                "trend": self._calculate_trend(values) if values else None
                            }
                        }
        except Exception as e:
            logging.debug(f"OECD error: {e}")
        
        return None
    
    def _map_to_oecd_code(self, indicator: str) -> Optional[str]:
        """Map indicator to OECD code"""
        # Direct match in indicators dict
        for key, value in self.indicators.items():
            if key == indicator.lower() or value.get("name", "").lower() == indicator.lower():
                if "oecd" in value:
                    return value["oecd"]
        
        # Common mappings
        mapping = {
            "gdp": "GDP",
            "inflation": "CPI",
            "cpi": "CPI",
            "unemployment": "UNRATE",
            "pmi": "PMI",
            "manufacturing pmi": "PMI",
            "business confidence": "BSRE",
            "consumer confidence": "CSCICP03",
            "economic sentiment": "ESD",
            "industrial production": "PRINTO",
            "trade balance": "B1_GE"
        }
        
        return mapping.get(indicator.lower())
    
    def _parse_oecd_response(self, data: Dict, indicator_code: str) -> List[Dict]:
        """Parse OECD SDMX-JSON response"""
        observations = []
        
        try:
            # This is a simplified parser - actual SDMX-JSON parsing is more complex
            # In production, use a proper SDMX library
            if "dataSets" in data and data["dataSets"]:
                series = data["dataSets"][0].get("series", {})
                
                # Get time periods from structure
                structure = data.get("structure", {})
                dimensions = structure.get("dimensions", {})
                observation_dim = dimensions.get("observation", [])
                
                # Find time dimension
                time_dim = None
                for dim in observation_dim:
                    if dim.get("keyPosition") == 1:  # Usually time is at position 1
                        time_dim = dim
                        break
                
                if time_dim and "values" in time_dim:
                    time_values = time_dim["values"]
                    
                    # Extract observations
                    for series_key, series_data in series.items():
                        observations_data = series_data.get("observations", {})
                        
                        for time_idx, obs_data in observations_data.items():
                            time_idx_int = int(time_idx)
                            if time_idx_int < len(time_values):
                                date_str = time_values[time_idx_int].get("name")
                                value = obs_data[0] if obs_data else None
                                
                                if value is not None:
                                    observations.append({
                                        "date": date_str,
                                        "value": float(value)
                                    })
        except Exception as e:
            logging.debug(f"Error parsing OECD response: {e}")
        
        # Sort by date descending
        observations.sort(key=lambda x: x["date"], reverse=True)
        return observations
    
    def _get_indicator_name(self, code: str) -> str:
        """Get human-readable indicator name from code"""
        names = {
            "GDP": "Gross Domestic Product",
            "CPI": "Consumer Price Index",
            "UNRATE": "Unemployment Rate",
            "PMI": "Purchasing Managers' Index",
            "BSRE": "Business Confidence",
            "CSCICP03": "Consumer Confidence",
            "ESD": "Economic Sentiment",
            "PRINTO": "Industrial Production",
            "B1_GE": "Trade Balance"
        }
        return names.get(code, code)
    
    async def _fetch_from_worldbank(self, indicator: str, country: str, options: Dict) -> Optional[Dict]:
        """
        Fetch data from World Bank API
        No API key required - free and open data
        """
        if not self.worldbank_enabled:
            return None
        
        # Map indicator to World Bank code
        wb_code = self._map_to_worldbank_code(indicator)
        if not wb_code:
            logging.debug(f"No World Bank code found for indicator: {indicator}")
            return None
        
        # World Bank API
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{wb_code}"
        
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
                        
                        if len(data) > 1 and data[1]:
                            observations = []
                            for item in data[1]:
                                if item.get("value") is not None:
                                    observations.append({
                                        "date": str(item.get("date")),
                                        "value": float(item.get("value"))
                                    })
                            
                            if not observations:
                                return None
                            
                            values = [o["value"] for o in observations]
                            
                            return {
                                "source": "worldbank",
                                "indicator": data[1][0].get("indicator", {}).get("value", indicator) if data[1] else indicator,
                                "country": country,
                                "frequency": "annual",  # World Bank data is typically annual
                                "observations": observations,
                                "latest_value": observations[0]["value"] if observations else None,
                                "latest_date": observations[0]["date"] if observations else None,
                                "statistics": {
                                    "mean": float(np.mean(values)) if values else None,
                                    "median": float(np.median(values)) if values else None,
                                    "min": float(min(values)) if values else None,
                                    "max": float(max(values)) if values else None,
                                    "trend": self._calculate_trend(values) if values else None
                                }
                            }
        except Exception as e:
            logging.debug(f"World Bank error: {e}")
        
        return None
    
    def _map_to_worldbank_code(self, indicator: str) -> Optional[str]:
        """Map indicator to World Bank code"""
        # Direct match in indicators dict
        for key, value in self.indicators.items():
            if key == indicator.lower() or value.get("name", "").lower() == indicator.lower():
                if "worldbank" in value:
                    return value["worldbank"]
        
        # Common mappings
        mapping = {
            "gdp": "NY.GDP.MKTP.CD",
            "gdp growth": "NY.GDP.MKTP.KD.ZG",
            "inflation": "FP.CPI.TOTL.ZG",
            "cpi": "FP.CPI.TOTL",
            "unemployment": "SL.UEM.TOTL.ZS",
            "trade balance": "NE.RSB.GNFS.CD",
            "exports": "NE.EXP.GNFS.CD",
            "imports": "NE.IMP.GNFS.CD"
        }
        
        return mapping.get(indicator.lower())
    
    async def _fetch_from_imf(self, indicator: str, country: str, options: Dict) -> Optional[Dict]:
        """
        Fetch data from IMF API
        No API key required - free and open data via SDMX 2.1 and 3.0 APIs
        
        IMF provides data through SDMX APIs. Data is available at:
        http://data.imf.org/
        """
        if not self.imf_enabled:
            return None
        
        # Map indicator to IMF code
        imf_code = self._map_to_imf_code(indicator)
        if not imf_code:
            logging.debug(f"No IMF code found for indicator: {indicator}")
            return None
        
        # Map country to IMF country code
        imf_country_code = self.imf_country_codes.get(country)
        if not imf_country_code:
            logging.debug(f"Country {country} not mapped to IMF country code")
            return None
        
        # IMF SDMX API endpoint
        # Using the JSON statistical API (SDMX-JSON)
        # Base URL for IMF's SDMX API
        base_url = "http://sdmx.imf.org/datastructure"
        
        # Alternative endpoint that might be more stable
        # The exact endpoint structure may need adjustment based on IMF's current API
        # This is a simplified implementation - in production, consult IMF API documentation
        
        # Using the Data API endpoint
        url = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
        
        # Construct the query
        # Format: /CompactData/{dataset}/{frequency}.{country}.{indicator}?startPeriod={year}&endPeriod={year}
        dataset = "IFS"  # International Financial Statistics (common dataset)
        frequency = "M"   # Monthly (A for annual, Q for quarterly)
        
        # Build the request URL
        query_url = f"{url}/CompactData/{dataset}/{frequency}.{imf_country_code}.{imf_code}?startPeriod={options.get('start_year', 2000)}&endPeriod={options.get('end_year', datetime.now().year)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(query_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse IMF SDMX-JSON response
                        observations = self._parse_imf_response(data, imf_code)
                        
                        if not observations:
                            return None
                        
                        values = [obs["value"] for obs in observations]
                        
                        return {
                            "source": "imf",
                            "indicator": self._get_imf_indicator_name(imf_code),
                            "country": country,
                            "frequency": self._get_imf_frequency(frequency),
                            "observations": observations[:options.get("max_obs", 100)],
                            "latest_value": observations[0]["value"] if observations else None,
                            "latest_date": observations[0]["date"] if observations else None,
                            "statistics": {
                                "mean": float(np.mean(values)) if values else None,
                                "median": float(np.median(values)) if values else None,
                                "min": float(min(values)) if values else None,
                                "max": float(max(values)) if values else None,
                                "trend": self._calculate_trend(values) if values else None
                            }
                        }
                    else:
                        # Try alternative endpoint or dataset
                        # IMF also provides data through their JSON API
                        alt_url = f"http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/M.{imf_country_code}.{imf_code}?startPeriod={options.get('start_year', 2000)}&endPeriod={options.get('end_year', datetime.now().year)}"
                        
                        async with session.get(alt_url) as alt_response:
                            if alt_response.status == 200:
                                alt_data = await alt_response.json()
                                observations = self._parse_imf_response(alt_data, imf_code)
                                
                                if observations:
                                    values = [obs["value"] for obs in observations]
                                    
                                    return {
                                        "source": "imf",
                                        "indicator": self._get_imf_indicator_name(imf_code),
                                        "country": country,
                                        "frequency": "monthly",
                                        "observations": observations[:options.get("max_obs", 100)],
                                        "latest_value": observations[0]["value"] if observations else None,
                                        "latest_date": observations[0]["date"] if observations else None,
                                        "statistics": {
                                            "mean": float(np.mean(values)) if values else None,
                                            "median": float(np.median(values)) if values else None,
                                            "min": float(min(values)) if values else None,
                                            "max": float(max(values)) if values else None,
                                            "trend": self._calculate_trend(values) if values else None
                                        }
                                    }
        except Exception as e:
            logging.debug(f"IMF error: {e}")
        
        return None
    
    def _map_to_imf_code(self, indicator: str) -> Optional[str]:
        """Map indicator to IMF code"""
        # Direct match in indicators dict
        for key, value in self.indicators.items():
            if key == indicator.lower() or value.get("name", "").lower() == indicator.lower():
                if "imf" in value:
                    return value["imf"]
        
        # Common mappings for IMF IFS (International Financial Statistics) codes
        mapping = {
            "gdp": "NGDPD",  # Gross Domestic Product, Current Prices
            "gdp growth": "NGDP_R",  # GDP, Constant Prices
            "inflation": "PCPI",  # Consumer Prices Index
            "cpi": "PCPI",  # Consumer Prices Index
            "core cpi": "PCPIP",  # Core Consumer Prices
            "ppi": "PPI",  # Producer Price Index
            "unemployment": "LUR",  # Unemployment Rate
            "labor force": "LFP",  # Labor Force Participation
            "pmi": "PMI",  # Purchasing Managers Index
            "manufacturing pmi": "PMI",
            "industrial production": "IIP",  # Industrial Production Index
            "trade balance": "BCA",  # Balance on Current Account
            "exports": "BX",  # Exports of Goods
            "imports": "BM",  # Imports of Goods
            "business confidence": "BCI",
            "consumer confidence": "CCI",
            "economic sentiment": "ESI"
        }
        
        return mapping.get(indicator.lower())
    
    def _parse_imf_response(self, data: Dict, indicator_code: str) -> List[Dict]:
        """Parse IMF SDMX-JSON response"""
        observations = []
        
        try:
            # IMF SDMX-JSON structure
            # This is a simplified parser - actual structure may vary
            if "CompactData" in data:
                compact_data = data["CompactData"]
                if "DataSet" in compact_data:
                    dataset = compact_data["DataSet"]
                    if "Series" in dataset:
                        series = dataset["Series"]
                        
                        # Handle both single series and list of series
                        if isinstance(series, dict):
                            series_list = [series]
                        else:
                            series_list = series
                        
                        for series_item in series_list:
                            if "Obs" in series_item:
                                obs_list = series_item["Obs"]
                                
                                # Handle both single observation and list
                                if isinstance(obs_list, dict):
                                    obs_list = [obs_list]
                                
                                for obs in obs_list:
                                    if "@OBS_VALUE" in obs:
                                        value = float(obs["@OBS_VALUE"])
                                        date = obs.get("@TIME_PERIOD", "")
                                        
                                        observations.append({
                                            "date": date,
                                            "value": value
                                        })
        except Exception as e:
            logging.debug(f"Error parsing IMF response: {e}")
        
        # Sort by date descending
        observations.sort(key=lambda x: x["date"], reverse=True)
        return observations
    
    def _get_imf_indicator_name(self, code: str) -> str:
        """Get human-readable indicator name from IMF code"""
        names = {
            "NGDPD": "GDP (Current Prices)",
            "NGDP_R": "Real GDP",
            "PCPI": "Consumer Price Index",
            "PCPIP": "Core CPI",
            "PPI": "Producer Price Index",
            "LUR": "Unemployment Rate",
            "LFP": "Labor Force Participation",
            "PMI": "Purchasing Managers' Index",
            "IIP": "Industrial Production",
            "BCA": "Current Account Balance",
            "BX": "Exports",
            "BM": "Imports",
            "BCI": "Business Confidence",
            "CCI": "Consumer Confidence",
            "ESI": "Economic Sentiment"
        }
        return names.get(code, code)
    
    def _get_imf_frequency(self, freq_code: str) -> str:
        """Convert IMF frequency code to human-readable"""
        frequencies = {
            "A": "annual",
            "Q": "quarterly",
            "M": "monthly",
            "D": "daily"
        }
        return frequencies.get(freq_code, "unknown")
    
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
                    "date": item.get("latest_date"),
                    "source": item.get("source")
                })
            elif any(word in indicator for word in ["cpi", "inflation", "price"]):
                composite["inflation_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date"),
                    "source": item.get("source")
                })
            elif any(word in indicator for word in ["unemployment", "payroll", "labor", "jobs"]):
                composite["labor_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date"),
                    "source": item.get("source")
                })
            elif any(word in indicator for word in ["sentiment", "confidence"]):
                composite["confidence_indicators"].append({
                    "name": item.get("indicator"),
                    "value": latest,
                    "date": item.get("latest_date"),
                    "source": item.get("source")
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
        This would typically use a dedicated economic calendar API
        """
        # Placeholder - would need integration with a calendar API
        # Examples: Investing.com, ForexFactory, etc.
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