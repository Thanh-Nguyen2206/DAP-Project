# Tắt GPU, chỉ dùng CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import thư viện
import streamlit as st

# Thiết lập giao diện Streamlit
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
from data_loader import AsyncDataLoader

# Try to import AI libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import custom modules
from prediction_models import train_prophet_model, train_randomforest_model, compare_models, create_prediction_chart
from technical_indicators import TechnicalIndicators
from auth_manager import AuthManager

# Try to import advanced_visualization
try:
    from advanced_visualization import AdvancedVisualizer
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False

# Try to import explainable_ai
try:
    from explainable_ai import ExplainableAI
    EXPLAINABLE_AI_AVAILABLE = True
    print("Explainable AI module loaded successfully")
except ImportError as e:
    EXPLAINABLE_AI_AVAILABLE = False
    print(f"Explainable AI module not available: {e}")

# Configuration Section
with st.sidebar:
    st.markdown("### Data Configuration")

# Set demo mode to True by default (no checkbox)
USE_REAL_DATA = False

# Initialize data loader with demo mode
if 'data_loader' not in st.session_state or st.session_state.get('demo_mode') != (not USE_REAL_DATA):
    st.session_state.data_loader = AsyncDataLoader(demo_mode=not USE_REAL_DATA)
    st.session_state.demo_mode = not USE_REAL_DATA

# Initialize authentication
auth = AuthManager.init_auth()

# Authentication Section
with st.sidebar:
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
        
    if not st.session_state.user_authenticated:
        st.markdown("###  Login Required")
        
        with st.form("login_form"):
            username = st.text_input("Username:", placeholder="Enter username")
            password = st.text_input("Password:", type="password", placeholder="Enter password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if auth._verify_credentials(username, password):
                    st.session_state.user_authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
                    
        # Show demo info
        st.info("Demo credentials:\n- Username: demo\n- Password: demo123")
        
    else:
        # Show logged in status  
        st.markdown(f"""
        **Mode**: {'Demo' if not USE_REAL_DATA else 'Live'}
        **User**: {st.session_state.get('username', 'demo')}
        **Status**: {'Admin' if st.session_state.username == 'admin' else 'User'}
        """)
        
        # Logout button
        if st.button(" Logout"):
            st.session_state.user_authenticated = False
            st.session_state.username = None
            st.rerun()

# Stop execution if not authenticated
if not st.session_state.user_authenticated:
    st.warning(" Please login to access the dashboard")
    st.stop()

# Main Application
st.title("Stock Analysis Dashboard")

# Data loader instance
data_loader = st.session_state.data_loader

# Initialize session state for persistent results
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# Initialize Advanced Visualizer
if 'advanced_viz' not in st.session_state and ADVANCED_VIZ_AVAILABLE:
    st.session_state.advanced_viz = AdvancedVisualizer()

# Initialize Explainable AI
if 'explainable_ai' not in st.session_state and EXPLAINABLE_AI_AVAILABLE:
    st.session_state.explainable_ai = ExplainableAI()

# Stock Selection in Sidebar
with st.sidebar:
    st.markdown("### Stock Selection")
    
    # Define available stocks based on mode
    available_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "JPM", "V", "WMT"]
    
    stock_symbol = st.selectbox(
        "Select Stock Symbol:",
        available_stocks,
        help="Choose a stock to analyze"
    )
    
    # Data Frequency Selection
    st.markdown("### Data Frequency")
    data_frequency = st.selectbox(
        "Select Data Frequency:",
        ["1m", "5m", "15m", "1h", "1d", "1w", "1M"],
        index=6,  # Default to "1M" (monthly)
        help="Choose the frequency of stock data"
    )
    
    # Date Range Selection
    st.markdown("###  Date Range")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2025-06-01").date(),
            help="Select start date for data"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            help="Select end date for data"
        )
    
    # Chart Type Selection
    st.markdown("### Chart Type")
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Line Chart", "Candlestick Chart", "OHLC Chart", "Area Chart"],
        index=1,  # Default to "Candlestick Chart"
        help="Choose the type of chart to display"
    )

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Technical Analysis", "Fundamental Analysis", "Price Prediction", "Advanced Visualization", "Explainable AI", "AI Chatbot"])

with tab1:
    st.header("Technical Analysis")
    
    try:
        # Fetch stock data
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
            df = result.get(stock_symbol) if result else None
        
        if df is not None and not df.empty:
            st.success(f"Successfully loaded {len(df)} data points for {stock_symbol}")
            
            # Basic Information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
                st.metric("Price Change", f"${price_change:.2f}", f"{(price_change/df['Close'].iloc[-2]*100):.2f}%" if len(df) > 1 else "N/A")
            with col3:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            with col4:
                st.metric("Data Points", len(df))
            
            # Technical Indicators
            # ti = TechnicalIndicators(df)  # Old way - not needed for static class
            
            # Moving Averages
            st.subheader("Moving Averages")
            ma_col1, ma_col2 = st.columns(2)
            
            with ma_col1:
                ma_short = st.slider("Short MA Period", 5, 50, 20)
            with ma_col2:
                ma_long = st.slider("Long MA Period", 20, 200, 50)
            
            # Create price chart with moving averages based on chart type
            fig = go.Figure()
            
            # Add main price chart based on selected type
            if chart_type == "Candlestick Chart":
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
            elif chart_type == "OHLC Chart":
                fig.add_trace(go.Ohlc(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
            elif chart_type == "Area Chart":
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0,100,255,0.2)'
                ))
            else:  # Line Chart
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
            
            # Add moving averages
            df[f'MA_{ma_short}'] = df['Close'].rolling(window=ma_short).mean()
            df[f'MA_{ma_long}'] = df['Close'].rolling(window=ma_long).mean()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'MA_{ma_short}'],
                mode='lines',
                name=f'MA {ma_short}',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'MA_{ma_long}'],
                mode='lines',
                name=f'MA {ma_long}',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title=f"{stock_symbol} {chart_type} with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI Indicator
            st.subheader("RSI (Relative Strength Index)")
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            
            # Calculate RSI using static method
            df_with_rsi = TechnicalIndicators.add_rsi(df, period=rsi_period)
            rsi = df_with_rsi['RSI'].dropna()
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # Add RSI levels
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig_rsi.update_layout(
                title=f"{stock_symbol} RSI ({rsi_period} periods)",
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100]),
                height=300
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Current RSI status
            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
            if current_rsi > 70:
                st.warning(f" RSI: {current_rsi:.2f} - Potentially Overbought")
            elif current_rsi < 30:
                st.info(f"🟢 RSI: {current_rsi:.2f} - Potentially Oversold")
            else:
                st.success(f"🟡 RSI: {current_rsi:.2f} - Neutral Zone")
            
            # Volume Analysis
            st.subheader("Volume Analysis")
            vol_ma = st.slider("Volume MA Period", 5, 50, 20)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            
            # Add volume moving average
            df[f'Vol_MA_{vol_ma}'] = df['Volume'].rolling(window=vol_ma).mean()
            fig_vol.add_trace(go.Scatter(
                x=df.index,
                y=df[f'Vol_MA_{vol_ma}'],
                mode='lines',
                name=f'Volume MA {vol_ma}',
                line=dict(color='red', width=2)
            ))
            
            fig_vol.update_layout(
                title=f"{stock_symbol} Volume Analysis",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
        else:
            st.error("Unable to fetch data for the selected stock")
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

with tab2:
    st.header("Fundamental Analysis")
    
    try:
        # Fetch stock data for fundamental analysis
        with st.spinner(f"Loading fundamental data for {stock_symbol}..."):
            result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
            df = result.get(stock_symbol) if result else None
        
        if df is not None and not df.empty:
            st.success(f" Data loaded successfully for {stock_symbol}")
            
            # Basic Statistics
            st.subheader(" Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Price Statistics:**")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"${df['Close'].mean():.2f}",
                        f"${df['Close'].median():.2f}",
                        f"${df['Close'].std():.2f}",
                        f"${df['Close'].min():.2f}",
                        f"${df['Close'].max():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            
            with col2:
                st.markdown("**Volume Statistics:**")
                vol_stats_df = pd.DataFrame({
                    'Metric': ['Mean Volume', 'Median Volume', 'Max Volume'],
                    'Value': [
                        f"{df['Volume'].mean():,.0f}",
                        f"{df['Volume'].median():,.0f}",
                        f"{df['Volume'].max():,.0f}"
                    ]
                })
                st.dataframe(vol_stats_df, hide_index=True)
            
            # Price Distribution
            st.subheader(" Price Distribution")
            
            fig_hist = px.histogram(
                df, x='Close', nbins=50,
                title=f"{stock_symbol} Price Distribution",
                labels={'Close': 'Price ($)', 'count': 'Frequency'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Returns Analysis
            st.subheader(" Returns Analysis")
            
            # Calculate returns
            df['Daily_Return'] = df['Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily returns histogram
                fig_returns = px.histogram(
                    df.dropna(), x='Daily_Return', nbins=50,
                    title="Daily Returns Distribution",
                    labels={'Daily_Return': 'Daily Return', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_returns, use_container_width=True)
            
            with col2:
                # Cumulative returns
                fig_cum_returns = go.Figure()
                fig_cum_returns.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Cumulative_Return'] * 100,
                    mode='lines',
                    name='Cumulative Return',
                    line=dict(color='green', width=2)
                ))
                fig_cum_returns.update_layout(
                    title="Cumulative Returns (%)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    height=400
                )
                st.plotly_chart(fig_cum_returns, use_container_width=True)
            
            # Risk Metrics
            st.subheader(" Risk Metrics")
            
            # Calculate risk metrics
            daily_returns = df['Daily_Return'].dropna()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(daily_returns, 5)
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.metric("Annual Volatility", f"{volatility:.2%}")
            with risk_col2:
                st.metric("Daily VaR (95%)", f"{var_95:.2%}")
            with risk_col3:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Recent Performance
            st.subheader(" Recent Performance")
            
            # Calculate performance metrics for different periods
            if len(df) >= 7:
                perf_1w = (df['Close'].iloc[-1] / df['Close'].iloc[-7] - 1) * 100
            else:
                perf_1w = 0
                
            if len(df) >= 30:
                perf_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1) * 100
            else:
                perf_1m = 0
                
            if len(df) >= 90:
                perf_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-90] - 1) * 100
            else:
                perf_3m = 0
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("1 Week", f"{perf_1w:.2f}%", delta=f"{perf_1w:.2f}%")
            with perf_col2:
                st.metric("1 Month", f"{perf_1m:.2f}%", delta=f"{perf_1m:.2f}%")
            with perf_col3:
                st.metric("3 Months", f"{perf_3m:.2f}%", delta=f"{perf_3m:.2f}%")
            
        else:
            st.error(" Unable to fetch data for fundamental analysis")
            
    except Exception as e:
        st.error(f" Error in fundamental analysis: {str(e)}")

with tab3:
    st.header(" Price Prediction")
    
    try:
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model:",
                ["Prophet", "Random Forest"],
                help="Choose the prediction model"
            )
        
        with col2:
            prediction_days = st.slider(
                "Prediction Days:",
                min_value=1,
                max_value=30,
                value=7,
                help="Number of days to predict"
            )
        
        # Generate prediction button
        if st.button(" Generate Prediction", type="primary"):
            with st.spinner(f"Generating {prediction_days}-day prediction for {stock_symbol}..."):
                # Fetch data
                result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
                df = result.get(stock_symbol) if result else None
                
                if df is not None and not df.empty:
                    # Check if we have enough data
                    min_required = 5  # Reduced from 30 to 5 for monthly data
                    if len(df) < min_required:
                        st.warning(f" Need at least {min_required} data points. Current: {len(df)}")
                        st.info(" Using available data points for prediction...")
                    
                    try:
                        # Generate prediction based on model type
                        if model_type == "Prophet":
                            model, forecast = train_prophet_model(df, prediction_days)
                            if model is not None and forecast is not None:
                                st.success(" Prophet model trained successfully!")
                                
                                # Create prediction chart
                                fig = create_prediction_chart(df, forecast, f"{stock_symbol} - Prophet Prediction")
                                
                                # Store results in session state
                                st.session_state.prediction_results = {
                                    'model_type': 'Prophet',
                                    'stock_symbol': stock_symbol,
                                    'df': df,
                                    'forecast': forecast,
                                    'fig': fig,
                                    'prediction_days': prediction_days
                                }
                                
                                # Store model for XAI analysis
                                st.session_state.last_prophet_model = model
                                
                            else:
                                st.error(" Failed to train Prophet model")
                        
                        else:  # Random Forest
                            model, predictions, dates = train_randomforest_model(df, prediction_days)
                            if model is not None and predictions is not None:
                                st.success(" Random Forest model trained successfully!")
                                
                                # Create prediction chart for Random Forest
                                fig = go.Figure()
                                
                                # Historical data
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['Close'],
                                    mode='lines',
                                    name='Historical Price',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Predictions
                                fig.add_trace(go.Scatter(
                                    x=dates,
                                    y=predictions,
                                    mode='lines+markers',
                                    name='Predicted Price',
                                    line=dict(color='red', width=2, dash='dash'),
                                    marker=dict(size=6)
                                ))
                                
                                # Set default date range from September 2023 to current
                                from datetime import datetime
                                start_display_date = datetime(2023, 9, 1)
                                end_display_date = datetime.now()
                                
                                fig.update_layout(
                                    title=f"{stock_symbol} - Random Forest Prediction",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    height=500,
                                    showlegend=True,
                                    xaxis=dict(
                                        range=[start_display_date, end_display_date],
                                        rangeslider=dict(visible=True),  # Add range slider for user to adjust
                                        rangeselector=dict(
                                            buttons=list([
                                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                                dict(step="all", label="All")
                                            ])
                                        )
                                    )
                                )
                                
                                # Store results in session state
                                st.session_state.prediction_results = {
                                    'model_type': 'Random Forest',
                                    'stock_symbol': stock_symbol,
                                    'df': df,
                                    'predictions': predictions,
                                    'dates': dates,
                                    'fig': fig,
                                    'prediction_days': prediction_days
                                }
                                
                                # Store model for XAI analysis
                                st.session_state.last_rf_model = model
                            
                            else:
                                st.error(" Failed to train Random Forest model")
                    
                    except Exception as model_error:
                        st.error(f" Model training error: {str(model_error)}")
                        st.info(" Try using a different model or check if there's enough data")
                
                else:
                    st.error(" Unable to fetch data for prediction")
        
        # Display stored prediction results if available
        if st.session_state.prediction_results is not None:
            pred_result = st.session_state.prediction_results
            
            # Debug information
            st.success(f" Found stored prediction results for {pred_result.get('model_type', 'Unknown')} model")
            
            # Display the chart
            st.plotly_chart(pred_result['fig'], use_container_width=True)
            
            # Show prediction summary
            st.subheader(" Prediction Summary")
            
            if pred_result['model_type'] == 'Prophet':
                try:
                    # Get last prediction values for Prophet
                    forecast = pred_result['forecast']
                    df = pred_result['df']
                    
                    # Debug info
                    st.write(f" **Prophet Model Summary for {pred_result['stock_symbol']}**")
                    
                    # Always show current price
                    current_price = df['Close'].iloc[-1]
                    
                    # Try to get future predictions
                    future_data = forecast[forecast['ds'] > df.index.max()]
                    
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    if not future_data.empty:
                        with pred_col2:
                            predicted_price = future_data['yhat'].iloc[-1]
                            st.metric("Predicted Price", f"${predicted_price:.2f}")
                        
                        with pred_col3:
                            price_change = predicted_price - current_price
                            change_pct = (price_change / current_price) * 100
                            st.metric("Expected Change", f"${price_change:.2f}", f"{change_pct:.2f}%")
                    else:
                        # Fallback: use last forecast value
                        with pred_col2:
                            predicted_price = forecast['yhat'].iloc[-1]
                            st.metric("Latest Forecast", f"${predicted_price:.2f}")
                        
                        with pred_col3:
                            price_change = predicted_price - current_price
                            change_pct = (price_change / current_price) * 100
                            st.metric("Price Difference", f"${price_change:.2f}", f"{change_pct:.2f}%")
                    
                    # Additional Prophet insights
                    st.markdown("** Prophet Model Insights:**")
                    col_insight1, col_insight2 = st.columns(2)
                    
                    with col_insight1:
                        st.info(f" Prediction Period: {pred_result['prediction_days']} days")
                    
                    with col_insight2:
                        trend_direction = " Upward" if predicted_price > current_price else " Downward"
                        st.info(f" Trend: {trend_direction}")
                        
                except Exception as e:
                    st.error(f" Error displaying Prophet summary: {str(e)}")
                    # Basic fallback
                    st.metric("Current Price", f"${pred_result['df']['Close'].iloc[-1]:.2f}")
            
            else:  # Random Forest
                try:
                    df = pred_result['df']
                    predictions = pred_result['predictions']
                    
                    # Debug info
                    st.write(f" **Random Forest Model Summary for {pred_result['stock_symbol']}**")
                    
                    current_price = df['Close'].iloc[-1]
                    predicted_price = predictions[-1]
                    
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with pred_col2:
                        st.metric("Predicted Price", f"${predicted_price:.2f}")
                    
                    with pred_col3:
                        price_change = predicted_price - current_price
                        change_pct = (price_change / current_price) * 100
                        st.metric("Expected Change", f"${price_change:.2f}", f"{change_pct:.2f}%")
                    
                    # Additional Random Forest insights
                    st.markdown("** Random Forest Model Insights:**")
                    col_insight1, col_insight2 = st.columns(2)
                    
                    with col_insight1:
                        st.info(f" Prediction Period: {pred_result['prediction_days']} days")
                    
                    with col_insight2:
                        trend_direction = " Upward" if predicted_price > current_price else " Downward"
                        st.info(f" Trend: {trend_direction}")
                    
                    # Show prediction confidence if available
                    if len(predictions) > 1:
                        avg_prediction = sum(predictions) / len(predictions)
                        st.info(f" Average Prediction: ${avg_prediction:.2f}")
                        
                except Exception as e:
                    st.error(f" Error displaying Random Forest summary: {str(e)}")
                    # Basic fallback
                    st.metric("Current Price", f"${pred_result['df']['Close'].iloc[-1]:.2f}")
        
        # Model comparison option
        st.markdown("---")
        st.subheader(" Model Comparison")
        
        if st.button(" Compare Models"):
            with st.spinner("Comparing Prophet vs Random Forest..."):
                result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
                df = result.get(stock_symbol) if result else None
                
                if df is not None and not df.empty and len(df) >= 5:
                    try:
                        comparison_results = compare_models(df, prediction_days)
                        
                        if comparison_results:
                            st.success(" Model comparison completed!")
                            
                            # Store comparison results in session state
                            st.session_state.comparison_results = {
                                'results': comparison_results,
                                'stock_symbol': stock_symbol,
                                'prediction_days': prediction_days
                            }
                            
                        else:
                            st.error(" Model comparison failed")
                    
                    except Exception as comp_error:
                        st.error(f" Comparison error: {str(comp_error)}")
                else:
                    st.error(" Insufficient data for model comparison")
        
        # Display stored comparison results if available
        if st.session_state.comparison_results is not None:
            comp_result = st.session_state.comparison_results
            
            # Show comparison metrics
            st.markdown("###  Performance Metrics")
            
            comparison_df = pd.DataFrame(comp_result['results'])
            st.dataframe(comparison_df, use_container_width=True)
            
            # Recommendations
            st.markdown("###  Recommendations")
            
            if len(comp_result['results']) >= 2:
                st.info(" **Both models trained successfully!**")
                st.info(" **Note**: Choose based on your preference for model type")
            elif len(comp_result['results']) == 1:
                model_name = comp_result['results'][0]['Model']
                st.info(f" **Available Model**: {model_name}")
            else:
                st.error(" No models were successfully trained")
    
    except Exception as e:
        st.error(f" Error in price prediction: {str(e)}")

with tab4:
    st.header(" Advanced Visualization")
    
    if not ADVANCED_VIZ_AVAILABLE:
        st.error(" Advanced Visualization module is not available. Please check installation.")
    else:
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Advanced Area Plot", "Waffle Chart", "Word Cloud", "Choropleth Map", "Correlation Heatmap", "Distribution Analysis"]
        )
        
        if viz_type == "Advanced Area Plot":
            st.subheader(" Advanced Multi-layer Area Plot")
            
            # Get current stock data from the session or fetch new data
            try:
                with st.spinner(f"Preparing advanced visualization for {stock_symbol}..."):
                    # Fetch fresh data for visualization
                    result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
                    df = result.get(stock_symbol) if result else None
                    
                    if df is not None and not df.empty:
                        # Calculate technical indicators using static methods
                        df_with_ma = TechnicalIndicators.add_moving_averages(df, [20, 50])
                        
                        # Prepare multi-layer data for advanced area plot
                        plot_data = pd.DataFrame({
                            'Date': df.index,
                            'Price': df['Close'],
                            'Volume_Scaled': df['Volume'] / df['Volume'].max() * df['Close'].max() * 0.3,
                            'MA20': df_with_ma['SMA_20'],
                            'MA50': df_with_ma['SMA_50']
                        })
                        
                        # Use the fixed Advanced Area Plot
                        fig = st.session_state.advanced_viz.create_advanced_area_plot(
                            plot_data, 
                            'Date',
                            ['Price', 'Volume_Scaled', 'MA20', 'MA50'],
                            f"{stock_symbol} Advanced Multi-Layer Analysis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success(" Advanced Area Plot created successfully!")
                        
                    else:
                        st.error(" No data available for visualization")
                        
            except Exception as e:
                st.error(f" Error creating Advanced Area Plot: {str(e)}")
                st.write("Debug info:", str(e))
        
        elif viz_type == "Waffle Chart":
            st.subheader(" Portfolio Composition Waffle Chart")
            
            # Sample portfolio data including current stock
            portfolio_data = {
                stock_symbol: 40,
                'AAPL': 25,
                'GOOGL': 20,
                'MSFT': 15
            }
            
            try:
                fig = st.session_state.advanced_viz.create_waffle_chart(
                    portfolio_data, 
                    f"Portfolio Allocation including {stock_symbol}"
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating waffle chart: {str(e)}")
        
        elif viz_type == "Word Cloud":
            st.subheader(" Stock Performance Word Cloud")
            
            try:
                # Fetch data for word cloud
                result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
                df = result.get(stock_symbol) if result else None
                
                if df is not None and not df.empty:
                    # Generate word cloud data from stock movements
                    word_data = {}
                    for i in range(1, min(len(df), 100)):
                        change = df['Close'].iloc[i] - df['Close'].iloc[i-1]
                        if change > 0:
                            word_data['bullish'] = word_data.get('bullish', 0) + abs(change)
                            word_data['growth'] = word_data.get('growth', 0) + abs(change) * 0.8
                            word_data['positive'] = word_data.get('positive', 0) + abs(change) * 0.6
                        else:
                            word_data['bearish'] = word_data.get('bearish', 0) + abs(change)
                            word_data['decline'] = word_data.get('decline', 0) + abs(change) * 0.8
                            word_data['correction'] = word_data.get('correction', 0) + abs(change) * 0.6
                    
                    word_data[stock_symbol] = max(word_data.values()) * 1.5 if word_data else 100
                    
                    fig = st.session_state.advanced_viz.create_wordcloud_visualization(
                        word_data, 
                        f"{stock_symbol} Performance Sentiment"
                    )
                    st.pyplot(fig)
                else:
                    st.error(" No data available for word cloud")
            except Exception as e:
                st.error(f"Error creating word cloud: {str(e)}")
        
        elif viz_type == "Choropleth Map":
            st.subheader(" Global Stock Market Performance Map")
            
            try:
                # Import required for streamlit map display
                import streamlit.components.v1 as components
                
                # Generate sample global market data (in real app, this would come from API)
                from advanced_visualization import generate_sample_geographic_data
                
                # Initialize data if not in session state
                if 'geographic_data' not in st.session_state:
                    st.session_state.geographic_data = generate_sample_geographic_data()
                
                geographic_data = st.session_state.geographic_data.copy()  # Make a copy to avoid mutations
                
                # Allow user to customize the data
                st.write("**Global Market Performance Data (%):**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    geographic_data['USA'] = st.slider("USA", -5.0, 10.0, geographic_data.get('USA', 2.5))
                    geographic_data['China'] = st.slider("China", -5.0, 10.0, geographic_data.get('China', 1.8))
                    geographic_data['Japan'] = st.slider("Japan", -5.0, 10.0, geographic_data.get('Japan', 1.2))
                    geographic_data['Germany'] = st.slider("Germany", -5.0, 10.0, geographic_data.get('Germany', 1.5))
                    geographic_data['UK'] = st.slider("UK", -5.0, 10.0, geographic_data.get('UK', 1.1))
                
                with col2:
                    geographic_data['France'] = st.slider("France", -5.0, 10.0, geographic_data.get('France', 0.9))
                    geographic_data['India'] = st.slider("India", -5.0, 10.0, geographic_data.get('India', 3.2))
                    geographic_data['Canada'] = st.slider("Canada", -5.0, 10.0, geographic_data.get('Canada', 1.3))
                    geographic_data['Australia'] = st.slider("Australia", -5.0, 10.0, geographic_data.get('Australia', 1.0))
                    geographic_data['Korea'] = st.slider("Korea", -5.0, 10.0, geographic_data.get('Korea', 2.1))
                
                with col3:
                    geographic_data['Brazil'] = st.slider("Brazil", -5.0, 10.0, geographic_data.get('Brazil', 2.8))
                    geographic_data['Mexico'] = st.slider("Mexico", -5.0, 10.0, geographic_data.get('Mexico', 1.7))
                    geographic_data['Italy'] = st.slider("Italy", -5.0, 10.0, geographic_data.get('Italy', 0.8))
                    geographic_data['Spain'] = st.slider("Spain", -5.0, 10.0, geographic_data.get('Spain', 0.7))
                    geographic_data['Netherlands'] = st.slider("Netherlands", -5.0, 10.0, geographic_data.get('Netherlands', 1.4))
                
                # Map type selection
                map_type = st.selectbox("Map Type", ["world", "usa"])
                
                # Create choropleth map with cache key to prevent duplicates
                cache_key = f"{map_type}_{hash(str(sorted(geographic_data.items())))}"
                
                if st.button("Generate New Map") or cache_key not in st.session_state:
                    map_obj = st.session_state.advanced_viz.create_choropleth_map(
                        geographic_data,
                        "Global Stock Market Performance (%)",
                        map_type
                    )
                    st.session_state[cache_key] = map_obj
                else:
                    map_obj = st.session_state[cache_key]
                
                if map_obj:
                    # Check if it's a Folium map or Plotly figure
                    if hasattr(map_obj, '_repr_html_'):
                        # Folium map - increased height and width by 50%
                        st.write("**Interactive Folium Map:**")
                        components.html(map_obj._repr_html_(), height=1200, width=1800)
                    else:
                        # Plotly figure - also increased height by 50%
                        st.write("**Plotly Choropleth Map:**") 
                        st.plotly_chart(map_obj, use_container_width=True, height=1200)
                
                # Display data table with validation
                st.write("**Data Summary:**")
                
                # Debug: check data before creating DataFrame
                valid_data = {}
                for country, value in geographic_data.items():
                    if value is not None and value != "":
                        valid_data[country] = value
                    else:
                        st.warning(f"Missing data for {country}, using default value 0.0")
                        valid_data[country] = 0.0
                
                data_df = pd.DataFrame(list(valid_data.items()), columns=['Country', 'Performance (%)'])
                data_df = data_df.sort_values('Performance (%)', ascending=False)
                data_df.reset_index(drop=True, inplace=True)
                st.dataframe(data_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating choropleth map: {str(e)}")
                st.info("Note: Choropleth maps require folium and geopandas libraries. Using Plotly fallback.")
        
        elif viz_type == "Correlation Heatmap":
            st.subheader(" Price & Indicators Correlation Matrix")
            
            try:
                # Fetch data for correlation analysis
                result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
                df = result.get(stock_symbol) if result else None
                
                if df is not None and not df.empty:
                    # Calculate indicators using static methods
                    df_with_indicators = TechnicalIndicators.add_rsi(df)
                    df_with_indicators = TechnicalIndicators.add_moving_averages(df_with_indicators, [20, 50])
                    
                    # Prepare correlation data
                    corr_data = pd.DataFrame({
                        'Close': df['Close'],
                        'Volume': df['Volume'],
                        'High': df['High'],
                        'Low': df['Low'],
                        'Open': df['Open'],
                        'RSI': df_with_indicators['RSI'],
                        'MA20': df_with_indicators['SMA_20'],
                        'MA50': df_with_indicators['SMA_50']
                    })
                    
                    fig = st.session_state.advanced_viz.create_correlation_heatmap(
                        corr_data, 
                        f"{stock_symbol} Correlation Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(" No data available for correlation analysis")
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {str(e)}")
        
        elif viz_type == "Distribution Analysis":
            st.subheader(" Price Distribution Analysis")
            
            try:
                # Fetch data for distribution analysis
                result = asyncio.run(data_loader.fetch_stock_data([stock_symbol], str(start_date), str(end_date), data_frequency))
                df = result.get(stock_symbol) if result else None
                
                if df is not None and not df.empty:
                    fig = st.session_state.advanced_viz.create_distribution_analysis(
                        df['Close'], 
                        f"{stock_symbol} Price Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(" No data available for distribution analysis")
            except Exception as e:
                st.error(f"Error creating distribution analysis: {str(e)}")

with tab5:
    st.header(" Explainable AI - Model Interpretability")
    
    if not EXPLAINABLE_AI_AVAILABLE:
        st.error(" Explainable AI module is not available. Please check installation.")
    else:
        st.info(" **Explainable AI Features**: Understand how ML models make predictions")
        
        # Simplified XAI demonstration
        explanation_type = st.selectbox(
            "Select Analysis Type",
            [" Model Analysis Demo", " Feature Importance", " Prediction Explanation", " Model Comparison"]
        )
        
        # Show slider and real-time explanation for Prediction Explanation
        instance_idx = 0
        if explanation_type == " Prediction Explanation":
            st.subheader("Individual Prediction Explanation")
            st.write("**Purpose**: Explain specific predictions for individual data points")
            st.info("ℹ️ **Real-time Analysis**: Slider updates explanation automatically")
            
            instance_idx = st.slider("Select data instance to explain", 0, 8, 0, key="xai_instance_slider")
            
            # Show current instance info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Instance", f"#{instance_idx}")
            with col2:
                st.metric("Analysis Type", "Individual")
            with col3:
                st.metric("Update Mode", "Real-time")
            
            # Auto-generate for Prediction Explanation when slider changes
            if f"xai_data_{explanation_type}" not in st.session_state:
                st.session_state[f"xai_data_{explanation_type}"] = None
                
            # Generate data once and store in session state
            with st.spinner("Preparing data for real-time explanation..."):
                try:
                    # Create demo data directly
                    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                    np.random.seed(42)  # For reproducible results
                    
                    # Generate synthetic stock data
                    base_price = 150
                    returns = np.random.normal(0.001, 0.02, len(dates))
                    prices = [base_price]
                    for ret in returns[1:]:
                        prices.append(prices[-1] * (1 + ret))
                    
                    demo_data = pd.DataFrame({
                        'Date': dates,
                        'Open': prices,
                        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                        'Close': prices,
                        'Volume': np.random.randint(1000000, 10000000, len(dates))
                    })
                    demo_data.set_index('Date', inplace=True)
                    
                    if len(demo_data) > 30:
                        # Create sample features
                        features_df = pd.DataFrame({
                            'Close': demo_data['Close'][-30:],
                            'Volume': demo_data['Volume'][-30:],
                            'High': demo_data['High'][-30:],
                            'Low': demo_data['Low'][-30:],
                            'Open': demo_data['Open'][-30:]
                        })
                        
                        # Add technical indicators using static methods with fallback
                        try:
                            from technical_indicators import TechnicalIndicators
                            demo_data_with_indicators = TechnicalIndicators.add_rsi(demo_data)
                            demo_data_with_indicators = TechnicalIndicators.add_moving_averages(demo_data_with_indicators, [20])
                            
                            if len(demo_data_with_indicators) > 30:
                                features_df['RSI'] = demo_data_with_indicators['RSI'][-30:].fillna(50)
                                features_df['SMA20'] = demo_data_with_indicators['SMA_20'][-30:].fillna(features_df['Close'])
                        except Exception as e:
                            # Fallback: Create simple technical indicators
                            features_df['RSI'] = np.random.uniform(30, 70, len(features_df))  # Random RSI
                            features_df['SMA20'] = features_df['Close'].rolling(window=min(20, len(features_df))).mean().fillna(features_df['Close'])
                        
                        features_df = features_df.dropna()
                        
                        if len(features_df) > 10:
                            # Create a simple model for demo
                            from sklearn.ensemble import RandomForestRegressor
                            from sklearn.model_selection import train_test_split
                            
                            X = features_df.drop('Close', axis=1)
                            y = features_df['Close']
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                            
                            # Train demo model
                            demo_model = RandomForestRegressor(n_estimators=10, random_state=42)
                            demo_model.fit(X_train, y_train)
                            
                            # Setup explainer
                            st.session_state.explainable_ai.setup_explainers(demo_model, X_train)
                            
                            # Store data for real-time use
                            st.session_state[f"xai_data_{explanation_type}"] = {
                                'X_test': X_test,
                                'demo_model': demo_model,
                                'X_train': X_train
                            }
                            
                except Exception as e:
                    st.error(f"Error preparing data: {str(e)}")
            
            # Real-time explanation generation
            if st.session_state[f"xai_data_{explanation_type}"] is not None:
                try:
                    data = st.session_state[f"xai_data_{explanation_type}"]
                    X_test = data['X_test']
                    
                    if instance_idx < len(X_test):
                        fig = st.session_state.explainable_ai.get_shap_waterfall(
                            X_test, instance_idx, f"{stock_symbol} Prediction Explanation (Instance {instance_idx})"
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Add detailed explanation for individual prediction
                            st.markdown("---")
                            st.markdown(f"### **Instance #{instance_idx} Prediction Analysis:**")
                            
                            # Get feature values for this instance
                            if instance_idx < len(X_test):
                                instance_data = X_test.iloc[instance_idx]
                                predicted_price = data['demo_model'].predict([instance_data])[0]
                                
                                st.success(f"""
                                **Individual Prediction Results:**
                                - **Predicted Price**: ${predicted_price:.2f}
                                - **Instance**: #{instance_idx} of {len(X_test)} test samples
                                - **Analysis**: Feature-by-feature contribution breakdown
                                """)
                                
                                # Show feature values for this instance
                                st.info("""
                                **Waterfall Chart Explanation:**
                                - **Base Value**: Average prediction across all data
                                - **Red Bars**: Features pushing price DOWN
                                - **Blue Bars**: Features pushing price UP  
                                - **Final Value**: Actual prediction for this instance
                                - **Arrow Flow**: Shows cumulative impact of each feature
                                """)
                                
                                # Display feature values table
                                st.markdown("#### Feature Values for This Instance:")
                                feature_table = pd.DataFrame({
                                    'Feature': instance_data.index,
                                    'Value': instance_data.values,
                                    'Type': ['Price' if 'Close' in f or 'Open' in f or 'High' in f or 'Low' in f 
                                            else 'Volume' if 'Volume' in f 
                                            else 'Technical Indicator' for f in instance_data.index]
                                })
                                st.dataframe(feature_table, use_container_width=True)
                        else:
                            st.error("Could not generate explanation")
                    else:
                        st.error(f"Instance {instance_idx} not available")
                except Exception as e:
                    st.error(f"Error generating real-time explanation: {str(e)}")
        else:
            # Regular button-based generation for other types
            if st.button(f" Generate {explanation_type}"):
                with st.spinner("Generating explanation..."):
                    try:
                        # Create demo data directly instead of fetching
                        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                        np.random.seed(42)  # For reproducible results
                        
                        # Generate synthetic stock data
                        base_price = 150
                        returns = np.random.normal(0.001, 0.02, len(dates))
                        prices = [base_price]
                        for ret in returns[1:]:
                            prices.append(prices[-1] * (1 + ret))
                        
                        demo_data = pd.DataFrame({
                            'Date': dates,
                            'Open': prices,
                            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                            'Close': prices,
                            'Volume': np.random.randint(1000000, 10000000, len(dates))
                        })
                        demo_data.set_index('Date', inplace=True)
                        
                        if len(demo_data) > 30:
                            # Create sample features
                            features_df = pd.DataFrame({
                                'Close': demo_data['Close'][-30:],
                                'Volume': demo_data['Volume'][-30:],
                                'High': demo_data['High'][-30:],
                                'Low': demo_data['Low'][-30:],
                                'Open': demo_data['Open'][-30:]
                            })
                            
                            # Add technical indicators using static methods with fallback
                            try:
                                from technical_indicators import TechnicalIndicators
                                demo_data_with_indicators = TechnicalIndicators.add_rsi(demo_data)
                                demo_data_with_indicators = TechnicalIndicators.add_moving_averages(demo_data_with_indicators, [20])
                                
                                if len(demo_data_with_indicators) > 30:
                                    features_df['RSI'] = demo_data_with_indicators['RSI'][-30:].fillna(50)
                                    features_df['SMA20'] = demo_data_with_indicators['SMA_20'][-30:].fillna(features_df['Close'])
                            except Exception as e:
                                # Fallback: Create simple technical indicators
                                st.warning(f"Using simplified technical indicators due to: {str(e)}")
                                features_df['RSI'] = np.random.uniform(30, 70, len(features_df))  # Random RSI
                                features_df['SMA20'] = features_df['Close'].rolling(window=min(20, len(features_df))).mean().fillna(features_df['Close'])
                            
                            features_df = features_df.dropna()
                            
                            if len(features_df) > 10:
                                # Create a simple model for demo
                                from sklearn.ensemble import RandomForestRegressor
                                from sklearn.model_selection import train_test_split
                                
                                X = features_df.drop('Close', axis=1)
                                y = features_df['Close']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                                
                                # Train demo model
                                demo_model = RandomForestRegressor(n_estimators=10, random_state=42)
                                demo_model.fit(X_train, y_train)
                                
                                # Setup explainer
                                st.session_state.explainable_ai.setup_explainers(demo_model, X_train)
                                
                                # Generate explanation based on type with distinct approaches
                                if explanation_type == " Model Analysis Demo":
                                    st.subheader("Model Analysis Demo")
                                    st.write("**Purpose**: Overview of ML model behavior and data characteristics")
                                    fig = st.session_state.explainable_ai.create_fallback_explanation(
                                        X_test, f"{stock_symbol} Model Analysis Demo"
                                    )
                                    
                                elif explanation_type == " Feature Importance":
                                    st.subheader("Global Feature Importance")
                                    st.write("**Purpose**: Overall ranking of feature importance across all predictions")
                                    
                                    # Create feature importance visualization
                                    feature_importance = demo_model.feature_importances_
                                    feature_names = X_train.columns
                                    
                                    # Create custom bar chart for feature importance
                                    import matplotlib.pyplot as plt
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Sort features by importance
                                    sorted_idx = np.argsort(feature_importance)[::-1]
                                    sorted_features = [feature_names[i] for i in sorted_idx]
                                    sorted_importance = feature_importance[sorted_idx]
                                    
                                    # Create horizontal bar chart
                                    bars = ax.barh(range(len(sorted_features)), sorted_importance)
                                    ax.set_yticks(range(len(sorted_features)))
                                    ax.set_yticklabels(sorted_features)
                                    ax.set_xlabel('Feature Importance Score')
                                    ax.set_title(f'{stock_symbol} - Global Feature Importance Analysis')
                                    
                                    # Add value labels on bars
                                    for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
                                        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                                               f'{importance:.3f}', va='center', ha='left')
                                    
                                    # Color bars based on importance
                                    for i, bar in enumerate(bars):
                                        if i < 2:  # Top 2 features
                                            bar.set_color('#e74c3c')  # Red for most important
                                        elif i < 4:  # Next 2 features  
                                            bar.set_color('#f39c12')  # Orange for moderate
                                        else:  # Remaining features
                                            bar.set_color('#3498db')  # Blue for less important
                                    
                                    plt.tight_layout()
                                    
                                else:  # Model Comparison
                                    st.subheader("Model Comparison Analysis")
                                    st.write("**Purpose**: Compare different aspects of model performance and explanations")
                                    fig = st.session_state.explainable_ai.compare_explanations(
                                        demo_model, X_train, X_test, f"{stock_symbol} Model Comparison"
                                    )
                                
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)  # Clean up memory
                                    
                                    # Add detailed explanation text based on type
                                    st.markdown("---")
                                    st.markdown("### **Analysis Explanation:**")
                                    
                                    if explanation_type == " Model Analysis Demo":
                                        st.success("""
                                        **Model Analysis Demo Results:**
                                        - **Data Overview**: Synthetic stock data with technical indicators
                                        - **Model Type**: Random Forest Regressor (10 trees)
                                        - **Features Used**: Close, Volume, High, Low, Open, RSI, SMA20
                                        - **Purpose**: Demonstrate ML model structure and data relationships
                                        """)
                                        
                                    elif explanation_type == " Feature Importance":
                                        st.success("""
                                        **Global Feature Importance Results:**
                                        - **Red bars**: Most critical features for price prediction
                                        - **Orange bars**: Moderately important features  
                                        - **Blue bars**: Less influential features
                                        - **Scores**: Higher values = greater impact on predictions
                                        - **Usage**: Helps identify which indicators drive model decisions
                                        """)
                                        
                                        # Add feature interpretation
                                        st.info("""
                                        **Feature Interpretation Guide:**
                                        - **Close/Open/High/Low**: Price-based features often most important
                                        - **Volume**: Trading activity indicator
                                        - **RSI**: Momentum oscillator (overbought/oversold)
                                        - **SMA20**: 20-day moving average trend indicator
                                        """)
                                        
                                    else:  # Model Comparison
                                        st.success("""
                                        **Model Comparison Results:**
                                        - **Comparison**: Different explanation methodologies
                                        - **Performance**: Model accuracy and prediction confidence
                                        - **Methods**: SHAP, LIME, and statistical approaches
                                        - **Purpose**: Validate model reliability across methods
                                        """)
                                else:
                                    st.error("Could not generate explanation - please try again")
                            else:
                                st.error("Not enough data for analysis")
                        else:
                            st.error("Not enough synthetic data generated for analysis")
                            
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
                        st.info(" This is a demo of Explainable AI features using synthetic data.")
        
        # Additional info
        st.markdown("---")
        st.markdown("""
        ###  **About Explainable AI (XAI)**
        
        **Why XAI matters in stock analysis:**
        - **Transparency**: Understand why the model made specific predictions
        - **Trust**: Build confidence in automated trading decisions  
        - **Debugging**: Identify potential model biases or errors
        - **Compliance**: Meet regulatory requirements for AI transparency
        
        **Available Methods:**
        - **SHAP**: Shows global and local feature importance
        - **LIME**: Explains individual predictions locally
        - **Feature Analysis**: Statistical analysis of input features
        
        *Note: This demo uses simplified models and sample data for demonstration purposes.*
        """)

with tab6:
    # Import and use enhanced chatbot
    try:
        from chatbot_enhanced import render_enhanced_chatbot
        render_enhanced_chatbot(f"Selected stock: {stock_symbol}, Data frequency: {data_frequency}")
    except ImportError:
        st.error(" Enhanced chatbot module not available")
        st.header(" Basic AI Chatbot")
        
        # Fallback to basic chatbot
        # Initialize chat history and typing state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'is_typing' not in st.session_state:
            st.session_state.is_typing = False
        
        # API Key Configuration
        api_key = st.text_input(
            "OpenAI API Key:", 
            type="password", 
            placeholder="Enter your OpenAI API key...",
            help="Get your API key from https://platform.openai.com/"
        )
        
        if api_key and OPENAI_AVAILABLE:
            # Basic chat interface
            st.info("ℹ Using basic chatbot. Enhanced version with Gemini and comparison available in enhanced module.")
        else:
            st.info(" Please configure your OpenAI API key to start chatting!")