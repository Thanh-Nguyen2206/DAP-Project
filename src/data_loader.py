import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import random

logging.basicConfig(level=logging.INFO)
from pathlib import Path
import joblib
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncDataLoader:
    def __init__(self, demo_mode=False):
        self.session = None
        self.cache_file = Path("cache/stock_data_cache.pkl")
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.cache = self._load_cache()
        self.demo_mode = demo_mode  # Allow configuration of demo mode

    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                cache = joblib.load(self.cache_file)
                # Clean expired entries
                now = pd.Timestamp.now()
                cache = {k: v for k, v in cache.items() 
                        if (now - v['timestamp']).total_seconds() < self.cache_ttl}
                return cache
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            joblib.dump(self.cache, self.cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        self.thread_pool.shutdown(wait=True)

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        age = (pd.Timestamp.now() - cache_entry['timestamp']).total_seconds()
        return age < self.cache_ttl

    async def fetch_stock_data(self, symbols: List[str], start_date: str, end_date: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        if self.demo_mode:
            return await self._fetch_demo_data(symbols, start_date, end_date, interval)
        else:
            return await self._fetch_live_data_with_retry(symbols, start_date, end_date, interval)

    async def fetch_multiple_stocks(self, symbols: List[str], start=None, end=None, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        results = {}
        for symbol in symbols:
            data = await self.fetch_stock_data([symbol], start_date=start, end_date=end, interval=interval)
            if data and symbol in data:
                results[symbol] = data[symbol]
            else:
                logger.error(f"Failed to fetch data for {symbol}")
        return results

    async def _fetch_live_data_with_retry(self, symbols: List[str], start_date: str, end_date: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Fetch live data with retry logic and rate limiting"""
        results = {}
        
        for symbol in symbols:
            max_retries = 3
            base_delay = 1
            
            for attempt in range(max_retries):
                try:
                    # Rate limiting - delay between requests
                    if attempt > 0:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying {symbol} after {delay:.2f}s delay (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    elif len(results) > 0:
                        # Small delay between symbols to avoid rate limiting
                        await asyncio.sleep(0.5)
                    
                    logger.info(f"Fetching live data for {symbol}")
                    
                    # Check cache first
                    cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
                    if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                        logger.info(f"Using cached data for {symbol}")
                        results[symbol] = self.cache[cache_key]['data']
                        break
                    
                    # Fetch from Yahoo Finance
                    stock = yf.Ticker(symbol)
                    data = stock.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        timeout=10
                    )
                    
                    if data.empty:
                        logger.warning(f"Empty data received for {symbol}")
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                        continue
                    
                    # Cache the data
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now().timestamp()
                    }
                    
                    results[symbol] = data
                    logger.info(f"Successfully fetched data for {symbol}")
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        logger.warning(f"Rate limited for {symbol}, attempt {attempt + 1}/{max_retries}")
                        if attempt == max_retries - 1:
                            logger.error(f"Rate limit exceeded for {symbol} after {max_retries} attempts")
                    elif "Expecting value" in error_msg:
                        logger.warning(f"JSON parse error for {symbol}, attempt {attempt + 1}/{max_retries}")
                        if attempt == max_retries - 1:
                            logger.error(f"JSON parse error for {symbol} after {max_retries} attempts")
                    else:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                    
                    if attempt == max_retries - 1:
                        # For final attempt failure, continue to next symbol instead of breaking
                        continue
        
        # Save cache after all requests
        self._save_cache()
        return results

    async def _fetch_demo_data(self, symbols: List[str], start_date: str, end_date: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Fetch demo data from local files"""
        results = {}
        demo_data_path = Path("demo_data")
        
        for symbol in symbols:
            try:
                # Try to load from demo data pickle files
                pickle_file = demo_data_path / f"{symbol}.pkl"
                if pickle_file.exists():
                    logger.info(f"Loading demo data for {symbol}")
                    data = joblib.load(pickle_file)
                    
                    # Apply resampling for different intervals
                    resampled_data = self._resample_data(data, interval)
                    results[symbol] = resampled_data
                else:
                    # Generate simple demo data if file doesn't exist
                    logger.info(f"Generating demo data for {symbol}")
                    demo_data = self._generate_demo_data(symbol, start_date, end_date)
                    resampled_data = self._resample_data(demo_data, interval)
                    results[symbol] = resampled_data
                    
            except Exception as e:
                logger.error(f"Error loading demo data for {symbol}: {e}")
                # Generate fallback demo data
                demo_data = self._generate_demo_data(symbol, start_date, end_date)
                resampled_data = self._resample_data(demo_data, interval)
                results[symbol] = resampled_data
        
        return results

    def _generate_demo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate simple demo data for testing"""
        try:
            # Extend date range to ensure enough data for monthly aggregation
            start_dt = pd.to_datetime(start_date) - timedelta(days=365)  # Add 1 year before
            end_dt = pd.to_datetime(end_date)
            
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            n_days = len(date_range)
            
            # Generate realistic stock price movements
            np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
            base_price = 100 + hash(symbol) % 500
            
            # Generate price movements with some volatility
            returns = np.random.normal(0.001, 0.02, n_days)
            prices = [base_price]
            
            for i in range(1, n_days):
                price = prices[-1] * (1 + returns[i])
                prices.append(max(price, 1))  # Prevent negative prices
            
            # Generate OHLC data
            data = []
            for i, (date, price) in enumerate(zip(date_range, prices)):
                daily_volatility = abs(np.random.normal(0, 0.015))
                high = price * (1 + daily_volatility)
                low = price * (1 - daily_volatility)
                
                # Ensure OHLC relationships are correct
                open_price = price + np.random.normal(0, price * 0.005)
                close_price = price + np.random.normal(0, price * 0.005)
                
                high = max(high, open_price, close_price, price)
                low = min(low, open_price, close_price, price)
                
                volume = int(np.random.uniform(1000000, 10000000))
                
                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(data, index=date_range)
            return df
            
        except Exception as e:
            logger.error(f"Error generating demo data: {e}")
            # Return minimal data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            data = {
                'Open': [100] * len(date_range),
                'High': [105] * len(date_range),
                'Low': [95] * len(date_range),
                'Close': [102] * len(date_range),
                'Volume': [1000000] * len(date_range)
            }
            return pd.DataFrame(data, index=date_range)


    def clear_cache(self, older_than: Optional[int] = None):
        """
        Clear the data cache with optional age filter
        """
        if older_than is None:
            self.cache.clear()
            logger.info("Cache fully cleared")
        else:
            now = pd.Timestamp.now()
            self.cache = {
                k: v for k, v in self.cache.items()
                if (now - v['timestamp']).total_seconds() <= older_than
            }
            logger.info(f"Cache entries older than {older_than} seconds cleared")
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a stock"""
        try:
            # Use demo data in demo mode
            if self.demo_mode:
                return {
                    'symbol': symbol,
                    'name': f'{symbol} Inc.',
                    'sector': 'Technology',
                    'industry': 'Software',
                    'market_cap': 1000000000,
                    'pe_ratio': 20.0,
                    'dividend_yield': 1.5,
                    'beta': 1.2,
                    'average_volume': 5000000,
                }

            # Try Yahoo Finance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Return a standardized info dictionary
            result = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'average_volume': info.get('averageVolume', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'roa': info.get('returnOnAssets', 0),
                'roe': info.get('returnOnEquity', 0),
            }
            
            # Convert None values to 0 or 'N/A'
            result = {
                k: (v if v is not None else (0 if isinstance(v, (int, float)) else 'N/A'))
                for k, v in result.items()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
            }

    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and trading hours"""
        try:
            if self.demo_mode:
                now = datetime.now()
                market_open = now.replace(hour=9, minute=30, second=0)
                market_close = now.replace(hour=16, minute=0, second=0)
                return {
                    'is_market_open': market_open <= now <= market_close,
                    'regular_market_price': 4000.0,  # Demo value
                    'market_open': market_open.strftime('%H:%M:%S'),
                    'market_close': market_close.strftime('%H:%M:%S'),
                    'trading_day': now.strftime('%Y-%m-%d')
                }

            # Use S&P 500 ETF as a proxy for market status
            spy = yf.Ticker("SPY")
            info = spy.info
            
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0)
            market_close = now.replace(hour=16, minute=0, second=0)
            
            return {
                'is_market_open': market_open <= now <= market_close,
                'regular_market_price': info.get('regularMarketPrice', 0),
                'regular_market_time': info.get('regularMarketTime', ''),
                'market_open': market_open.strftime('%H:%M:%S'),
                'market_close': market_close.strftime('%H:%M:%S'),
                'trading_day': now.strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {
                'error': str(e),
                'is_market_open': False
            }

    def _resample_data(self, data: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample data to different frequency intervals"""
        try:
            if data.empty:
                return data
            
            # Mapping from interval codes to pandas resample frequency
            interval_map = {
                '1m': '1min',
                '5m': '5min', 
                '15m': '15min',
                '1h': '1H',
                '1d': '1D',
                '1w': '1W',
                '1M': '1M'
            }
            
            freq = interval_map.get(interval, '1D')
            
            # If it's already daily data and requesting daily, return as-is
            if freq == '1D':
                return data
                
            # For higher frequency than daily, we need to interpolate/simulate
            if freq in ['1min', '5min', '15min', '1H']:
                # For demo purposes, create intraday data by interpolating
                return self._simulate_intraday_data(data, freq)
            
            # For weekly data, resample
            if freq == '1W':
                resampled = data.resample(freq).agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                return resampled
            
            # For monthly data, resample
            if freq == '1M':
                resampled = data.resample(freq).agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                return resampled
                
            return data
            
        except Exception as e:
            logger.warning(f"Error resampling data: {e}")
            return data
    
    def _simulate_intraday_data(self, daily_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Simulate intraday data from daily data for demo purposes"""
        try:
            if daily_data.empty:
                return daily_data
            
            # Take a subset of daily data for performance
            sample_data = daily_data.tail(30)  # Last 30 days
            
            simulated_data = []
            
            for date, row in sample_data.iterrows():
                # Number of intervals per day
                intervals_per_day = {
                    '1min': 390,  # 6.5 hours * 60 minutes
                    '5min': 78,   # 6.5 hours * 12 intervals
                    '15min': 26,  # 6.5 hours * 4 intervals  
                    '1H': 7       # 6.5 hours
                }.get(freq, 7)
                
                # Create intraday timestamps
                base_date = pd.to_datetime(date).date()
                start_time = pd.Timestamp.combine(base_date, pd.Timestamp('09:30:00').time())
                
                for i in range(intervals_per_day):
                    if freq == '1min':
                        timestamp = start_time + pd.Timedelta(minutes=i)
                    elif freq == '5min':
                        timestamp = start_time + pd.Timedelta(minutes=i*5)
                    elif freq == '15min':
                        timestamp = start_time + pd.Timedelta(minutes=i*15)
                    elif freq == '1H':
                        timestamp = start_time + pd.Timedelta(hours=i)
                    
                    # Simulate price movement within the day
                    price_range = row['High'] - row['Low']
                    noise_factor = np.random.uniform(-0.5, 0.5)
                    
                    # Calculate OHLC for this interval
                    base_price = row['Low'] + (price_range * i / intervals_per_day)
                    noise = price_range * 0.1 * noise_factor
                    
                    interval_open = base_price + noise
                    interval_close = base_price + noise + (row['Close'] - row['Open']) / intervals_per_day
                    interval_high = max(interval_open, interval_close) + abs(noise) * 0.5
                    interval_low = min(interval_open, interval_close) - abs(noise) * 0.5
                    
                    simulated_data.append({
                        'Open': interval_open,
                        'High': interval_high,
                        'Low': interval_low, 
                        'Close': interval_close,
                        'Volume': row['Volume'] / intervals_per_day
                    })
            
            # Create DataFrame with proper index
            if simulated_data:
                result_df = pd.DataFrame(simulated_data)
                # Create proper datetime index
                start_date = sample_data.index[0]
                periods = len(simulated_data)
                if freq == '1min':
                    freq_str = '1min'
                elif freq == '5min':
                    freq_str = '5min'
                elif freq == '15min':
                    freq_str = '15min'
                elif freq == '1H':
                    freq_str = '1H'
                
                result_df.index = pd.date_range(start=start_date, periods=periods, freq=freq_str)
                return result_df
            
            return daily_data
            
        except Exception as e:
            logger.warning(f"Error simulating intraday data: {e}")
            return daily_data