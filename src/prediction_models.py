"""
Simple prediction models for stock price forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def train_prophet_model(data, prediction_days=7):
    """
    Simple Prophet-like model using moving averages
    """
    try:
        # Prepare data
        df = data.copy()
        df = df.reset_index()
        
        # Handle date column properly
        if 'Date' in df.columns:
            df['ds'] = pd.to_datetime(df['Date'])
        else:
            # Reset index to get dates
            df = df.reset_index()
            if 'Date' in df.columns:
                df['ds'] = pd.to_datetime(df['Date'])
            else:
                # Use index as dates - more robust handling
                try:
                    df['ds'] = pd.to_datetime(df.index, errors='coerce')
                    # If conversion failed, create date range
                    if df['ds'].isna().all():
                        df['ds'] = pd.date_range(end=pd.Timestamp.now().date(), periods=len(df), freq='D')
                except Exception:
                    # Fallback: create date range
                    df['ds'] = pd.date_range(end=pd.Timestamp.now().date(), periods=len(df), freq='D')
        
        df['y'] = df['Close']
        
        # Ensure we have enough data
        if len(df) < 5:  # Reduced from 30 to 5 for monthly data
            print(f"Insufficient data for Prophet model: {len(df)} points, need at least 5")
            return None, None
        
        # Simple trend calculation - adjust window size based on data length
        window_size = max(3, min(10, len(df)//2))  # Adaptive window size
        df['trend'] = df['y'].rolling(window=window_size).mean()
        df = df.dropna()  # Remove NaN values from rolling calculation
        
        if len(df) < 3:  # Reduced from 10 to 3
            print(f"Not enough data after trend calculation: {len(df)} points")
            return None, None
        
        # Generate future dates
        last_date = df['ds'].iloc[-1]
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=prediction_days, freq='D')
        
        # Simple prediction using trend
        last_trend = df['trend'].iloc[-1]
        if pd.isna(last_trend):
            last_trend = df['y'].iloc[-1]
        
        # Calculate trend slope more safely
        trend_window = min(10, len(df))
        if len(df) >= trend_window:
            trend_slope = (df['trend'].iloc[-1] - df['trend'].iloc[-trend_window]) / trend_window
        else:
            trend_slope = 0
        
        # Handle NaN trend slope
        if pd.isna(trend_slope):
            trend_slope = 0
        
        predictions = []
        for i, date in enumerate(future_dates):
            pred_value = last_trend + (trend_slope * (i + 1))
            # Add some random variation to make it more realistic
            variation = np.random.normal(0, abs(pred_value) * 0.02)
            pred_value += variation
            
            predictions.append({
                'ds': date,
                'yhat': pred_value,
                'yhat_lower': pred_value * 0.95,
                'yhat_upper': pred_value * 1.05
            })
        
        pred_df = pd.DataFrame(predictions)
        
        # Create complete forecast with historical + predicted data
        forecast = pd.DataFrame({
            'ds': pd.concat([df['ds'], pred_df['ds']], ignore_index=True),
            'yhat': pd.concat([df['y'], pred_df['yhat']], ignore_index=True),
            'yhat_lower': pd.concat([df['y'] * 0.98, pred_df['yhat_lower']], ignore_index=True), 
            'yhat_upper': pd.concat([df['y'] * 1.02, pred_df['yhat_upper']], ignore_index=True)
        })
        
        # Return mock model and complete forecast
        model = "simple_prophet_model"
        return model, forecast
        
    except Exception as e:
        print(f"Error in Prophet model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_randomforest_model(data, prediction_days=7):
    """
    Random Forest model for price prediction
    """
    try:
        # Prepare features
        df = data.copy()
        df = df.reset_index()
        
        # Ensure we have enough data
        if len(df) < 5:  # Reduced from 30 to 5 for monthly data
            print(f"Insufficient data for Random Forest model: {len(df)} points, need at least 5")
            return None, None
        
        # Technical indicators as features - adaptive windows
        window_5 = max(2, min(5, len(df)//4))
        window_20 = max(3, min(10, len(df)//2))
        
        df['sma_5'] = df['Close'].rolling(window_5).mean()
        df['sma_20'] = df['Close'].rolling(window_20).mean()
        df['rsi'] = calculate_rsi(df['Close'], window=max(2, min(14, len(df)//3)))
        df['volatility'] = df['Close'].rolling(window_5).std()
        
        # Lag features - only use lag 1 for small datasets
        max_lag = min(3, len(df)//3)
        for lag in range(1, max_lag + 1):
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 3:  # Reduced from 20 to 3
            print(f"Not enough data after feature engineering: {len(df)} points")
            return None, None
        
        # Prepare training data
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility']
        # Only add lag features that exist
        for lag in range(1, max_lag + 1):
            col_name = f'close_lag_{lag}'
            if col_name in df.columns:
                feature_cols.append(col_name)
        
        # Check for NaN values in features and fill them
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
            else:
                # Remove missing columns from feature list
                feature_cols = [c for c in feature_cols if c != col]
        
        X = df[feature_cols].values
        y = df['Close'].values
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Generate predictions
        predictions = []
        last_features = X[-1].copy()
        
        # Handle date generation
        if 'Date' in df.columns:
            last_date = pd.to_datetime(df['Date'].iloc[-1])
        elif hasattr(df.index[-1], 'date'):
            last_date = pd.to_datetime(df.index[-1])
        else:
            last_date = pd.Timestamp.now().date()
        
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=prediction_days, freq='D')
        
        for i in range(prediction_days):
            pred = model.predict([last_features])[0]
            predictions.append(pred)
            
            # Update features for next prediction
            if len(last_features) >= 3:
                last_features[4:] = last_features[3:-1]  # Shift lag features
                last_features[3] = pred  # Update close_lag_1
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': [p * 0.92 for p in predictions],
            'yhat_upper': [p * 1.08 for p in predictions]
        })
        
        # Return model, predictions (values), and dates to match app expectation
        return model, predictions, future_dates
        
    except Exception as e:
        print(f"Error in Random Forest model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    except:
        return pd.Series([50] * len(prices), index=prices.index)


def compare_models(data, prediction_days=7):
    """
    Compare different models performance
    """
    try:
        if data is None or data.empty:
            return None
        
        # Train both models
        prophet_model, prophet_forecast = train_prophet_model(data, prediction_days)
        rf_model, rf_predictions, rf_dates = train_randomforest_model(data, prediction_days)
        
        comparison = []
        
        # Add Prophet results if successful
        if prophet_model is not None and prophet_forecast is not None:
            # Get future predictions only
            future_data = prophet_forecast[prophet_forecast['ds'] > data.index.max()]
            if not future_data.empty:
                comparison.append({
                    'Model': 'Prophet',
                    'Next Day Prediction': f"${future_data['yhat'].iloc[0]:.2f}",
                    'Week Average': f"${future_data['yhat'].mean():.2f}",
                    'Confidence Range': f"${future_data['yhat_lower'].mean():.2f} - ${future_data['yhat_upper'].mean():.2f}",
                    'MAE': 'N/A',  # Would need historical validation for real MAE
                    'RMSE': 'N/A'
                })
        
        # Add Random Forest results if successful  
        if rf_model is not None and rf_predictions is not None:
            comparison.append({
                'Model': 'Random Forest',
                'Next Day Prediction': f"${rf_predictions[0]:.2f}",
                'Week Average': f"${np.mean(rf_predictions):.2f}",
                'Confidence Range': f"${np.mean(rf_predictions) * 0.92:.2f} - ${np.mean(rf_predictions) * 1.08:.2f}",
                'MAE': 'N/A',  # Would need historical validation for real MAE
                'RMSE': 'N/A'
            })
        
        if not comparison:
            return None
            
        return comparison
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_prediction_chart(data, predictions, title="Stock Price Predictions"):
    """
    Create interactive chart with predictions
    Can handle either predictions_dict (multiple models) or single prediction DataFrame
    """
    try:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(title,)
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Historical Price',
                line=dict(color='blue')
            )
        )
        
        # Handle different prediction formats
        if isinstance(predictions, dict):
            # Multiple models case (predictions_dict)
            colors = ['red', 'green', 'orange', 'purple']
            for i, (model_name, (model, pred_data)) in enumerate(predictions.items()):
                if pred_data is not None and not pred_data.empty:
                    color = colors[i % len(colors)]
                    
                    # Prediction line
                    fig.add_trace(
                        go.Scatter(
                            x=pred_data['ds'],
                            y=pred_data['yhat'],
                            name=f'{model_name} Prediction',
                            line=dict(color=color, dash='dash')
                        )
                    )
                    
                    # Confidence band
                    fig.add_trace(
                        go.Scatter(
                            x=list(pred_data['ds']) + list(pred_data['ds'][::-1]),
                            y=list(pred_data['yhat_upper']) + list(pred_data['yhat_lower'][::-1]),
                            fill='tonexty',
                            fillcolor=f'rgba(255,0,0,0.2)' if color == 'red' else f'rgba(0,255,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{model_name} Confidence',
                            showlegend=False
                        )
                    )
        else:
            # Single prediction DataFrame case
            if predictions is not None and not predictions.empty:
                # Prediction line
                fig.add_trace(
                    go.Scatter(
                        x=predictions['ds'],
                        y=predictions['yhat'],
                        name='Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Confidence band
                fig.add_trace(
                    go.Scatter(
                        x=list(predictions['ds']) + list(predictions['ds'][::-1]),
                        y=list(predictions['yhat_upper']) + list(predictions['yhat_lower'][::-1]),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Band',
                        showlegend=False
                    )
                )
        
        # Set default date range from September 2023 to current
        from datetime import datetime
        start_display_date = datetime(2023, 9, 1)
        end_display_date = datetime.now()
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            hovermode='x unified',
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
        
        return fig
        
    except Exception as e:
        print(f"Error creating prediction chart: {e}")
        import traceback
        traceback.print_exc()
        return None