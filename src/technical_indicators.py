import pandas as pd
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    import ta
import plotly.graph_objects as go
import streamlit as st
from typing import Tuple, Optional, List, Dict, Any

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list = [20, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Averages"""
        df = df.copy()
        for period in periods:
            if TALIB_AVAILABLE:
                df[f'SMA_{period}'] = talib.SMA(df['Close'].values, timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(df['Close'].values, timeperiod=period)
            else:
                # Use ta library as fallback
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = df.copy()
        if TALIB_AVAILABLE:
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                df['Close'].values, 
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev
            )
        else:
            # Use ta library as fallback
            bb_indicator = ta.volatility.BollingerBands(df['Close'], window=period, window_dev=std_dev)
            df['BB_Upper'] = bb_indicator.bollinger_hband()
            df['BB_Middle'] = bb_indicator.bollinger_mavg()
            df['BB_Lower'] = bb_indicator.bollinger_lband()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        df = df.copy()
        if TALIB_AVAILABLE:
            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=period)
        else:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=period)
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        df = df.copy()
        if TALIB_AVAILABLE:
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
                df['Close'].values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
        else:
            macd_indicator = ta.trend.MACD(df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
            df['MACD'] = macd_indicator.macd()
            df['MACD_Signal'] = macd_indicator.macd_signal()
            df['MACD_Hist'] = macd_indicator.macd_diff()
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df = df.copy()
        if TALIB_AVAILABLE:
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                fastk_period=k,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=d,
                slowd_matype=0
            )
        else:
            stoch_indicator = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=k, smooth_window=d)
            df['STOCH_K'] = stoch_indicator.stoch()
            df['STOCH_D'] = stoch_indicator.stoch_signal()
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        df = df.copy()
        if TALIB_AVAILABLE:
            df['ATR'] = talib.ATR(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )
        else:
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
        return df
    
    @staticmethod
    def add_fibonacci_levels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Fibonacci Retracement Levels"""
        df = df.copy()
        high = df['High'].rolling(period).max()
        low = df['Low'].rolling(period).min()
        diff = high - low
        
        df['Fib_0'] = low
        df['Fib_236'] = low + 0.236 * diff
        df['Fib_382'] = low + 0.382 * diff
        df['Fib_500'] = low + 0.500 * diff
        df['Fib_618'] = low + 0.618 * diff
        df['Fib_100'] = high
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        df = df.copy()
        
        # On-Balance Volume
        if TALIB_AVAILABLE:
            df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)
            # Money Flow Index
            df['MFI'] = talib.MFI(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                df['Volume'].values,
                timeperiod=14
            )
            # Volume SMA
            df['Volume_SMA'] = talib.SMA(df['Volume'].values, timeperiod=20)
        else:
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame"""
        df = df.copy()
        
        # Add basic indicators
        df = self.add_moving_averages(df)
        df = self.add_bollinger_bands(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_stochastic(df)
        df = self.add_atr(df)
        df = self.add_fibonacci_levels(df)
        df = self.add_volume_indicators(df)
        
        # Add custom indicators
        df['Trend'] = np.where(df['Close'] > df['SMA_50'], 1, -1)
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['Momentum'] = df['Close'].pct_change(periods=20)
        
        return df
    
    @staticmethod
    def get_signals(df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate trading signals based on technical indicators
        
        Returns:
        --------
        Dict with signal types:
            - trend: 'up', 'down', or 'neutral'
            - strength: 'strong', 'medium', or 'weak'
            - volume: 'high', 'normal', or 'low'
            - volatility: 'high', 'normal', or 'low'
        """
        last_row = df.iloc[-1]
        
        # Trend signals
        trend = 'neutral'
        if last_row['Close'] > last_row['SMA_50'] and last_row['MACD'] > last_row['MACD_Signal']:
            trend = 'up'
        elif last_row['Close'] < last_row['SMA_50'] and last_row['MACD'] < last_row['MACD_Signal']:
            trend = 'down'
            
        # Strength signals
        strength = 'medium'
        if last_row['RSI'] > 70 or last_row['RSI'] < 30:
            strength = 'strong'
        elif 40 <= last_row['RSI'] <= 60:
            strength = 'weak'
            
        # Volume signals
        volume = 'normal'
        if last_row['Volume_Ratio'] > 1.5:
            volume = 'high'
        elif last_row['Volume_Ratio'] < 0.5:
            volume = 'low'
            
        # Volatility signals
        volatility = 'normal'
        if last_row['ATR'] > last_row['ATR'].mean() * 1.5:
            volatility = 'high'
        elif last_row['ATR'] < last_row['ATR'].mean() * 0.5:
            volatility = 'low'
            
        return {
            'trend': trend,
            'strength': strength,
            'volume': volume,
            'volatility': volatility
        }
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators"""
        df = df.copy()
        
        # Conversion Line (Tenkan-sen)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Conversion_Line'] = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen)
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Base_Line'] = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A)
        df['Leading_Span_A'] = ((df['Conversion_Line'] + df['Base_Line']) / 2).shift(26)
        
        # Leading Span B (Senkou Span B)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Leading_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Lagging Span (Chikou Span)
        df['Lagging_Span'] = df['Close'].shift(-26)
        
        return df

# Hàm tính Ichimoku Cloud
def calculate_ichimoku(data):
    """
    Tính các thành phần của Ichimoku Cloud.
    
    Parameters:
    - data: DataFrame với dữ liệu giá OHLC
    
    Returns:
    - Dictionary chứa các đường của Ichimoku
    """
    try:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = data['High'].rolling(window=9).max()
        nine_period_low = data['Low'].rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        twenty_six_period_high = data['High'].rolling(window=26).max()
        twenty_six_period_low = data['Low'].rolling(window=26).min()
        kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        fifty_two_period_high = data['High'].rolling(window=52).max()
        fifty_two_period_low = data['Low'].rolling(window=52).min()
        senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

        # Chikou Span (Lagging Span): Current closing price shifted back 26 periods
        chikou_span = data['Close'].shift(-26)

        return {
            'Tenkan-sen': tenkan_sen,
            'Kijun-sen': kijun_sen,
            'Senkou Span A': senkou_span_a,
            'Senkou Span B': senkou_span_b,
            'Chikou Span': chikou_span
        }
    except Exception as e:
        st.error(f"Error calculating Ichimoku Cloud: {str(e)}")
        return None

    def plot_technical_analysis(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Plot technical analysis indicators
        """
        df = df.copy()
        df = self.add_all_indicators(df)
        
        signals = self.get_signals(df)
        latest_close = df['Close'].iloc[-1]
        
        print("\nTechnical Analysis Results:")
        print(f"Current Price: ${latest_close:.2f}")
        print(f"Trend: {signals['trend']}")
        print(f"Strength: {signals['strength']}")
        print(f"Volume: {signals['volume']}")
        print(f"Volatility: {signals['volatility']}")
        
        return df
    def __init__(self):
        """Initialize TechnicalIndicators class"""
        pass  # No initialization needed at this point

# Hàm vẽ Ichimoku Cloud
def plot_ichimoku(fig, ichimoku, data_index):
    """
    Thêm các thành phần của Ichimoku Cloud vào biểu đồ.
    
    Parameters:
    - fig: Plotly Figure object
    - ichimoku: Dictionary chứa các đường Ichimoku
    - data_index: Index của dữ liệu
    """
    # Vẽ Tenkan-sen
    fig.add_trace(go.Scatter(
        x=data_index,
        y=ichimoku['Tenkan-sen'],
        name='Tenkan-sen',
        line=dict(color='red'),
        visible='legendonly'
    ))
    
    # Vẽ Kijun-sen
    fig.add_trace(go.Scatter(
        x=data_index,
        y=ichimoku['Kijun-sen'],
        name='Kijun-sen',
        line=dict(color='blue'),
        visible='legendonly'
    ))
    
    # Vẽ Senkou Span A và B (Cloud)
    fig.add_trace(go.Scatter(
        x=data_index,
        y=ichimoku['Senkou Span A'],
        name='Senkou Span A',
        line=dict(color='green'),
        fill=None,
        visible='legendonly'
    ))
    
    fig.add_trace(go.Scatter(
        x=data_index,
        y=ichimoku['Senkou Span B'],
        name='Senkou Span B',
        line=dict(color='red'),
        fill='tonexty',  # Fill to previous trace
        visible='legendonly'
    ))
    
    # Vẽ Chikou Span
    fig.add_trace(go.Scatter(
        x=data_index,
        y=ichimoku['Chikou Span'],
        name='Chikou Span',
        line=dict(color='purple'),
        visible='legendonly'
    ))