# Stock Market Analysis with AI Integration

A comprehensive web-based stock market analysis platform built with Python, Streamlit, and AI technologies for intelligent investment decision-making.

## Features

### 6 Main Analysis Modules

1. **Technical Analysis**
   - Candlestick charts with interactive controls
   - RSI, MACD, Bollinger Bands indicators
   - Moving averages (MA20, MA50, MA200)
   - Automated trend analysis

2. **Fundamental Analysis**
   - Company information and financial metrics
   - P/E Ratio, Market Cap, Revenue analysis
   - Sector comparison tools
   - Key financial indicators

3. **Price Prediction**
   - Prophet model for long-term forecasting
   - Random Forest for multi-factor analysis
   - LSTM neural network for deep learning
   - 1-30 day prediction horizons

4. **Advanced Visualization**
   - Choropleth maps for global market performance
   - 3D correlation analysis
   - Word clouds and distribution plots
   - Interactive data exploration

5. **Explainable AI**
   - SHAP waterfall charts
   - Feature importance analysis
   - Model comparison tools
   - Real-time prediction explanations

6. **AI Chatbot**
   - Dual AI engines (OpenAI GPT-4 + Google Gemini 2.0)
   - Financial advisory capabilities
   - Technical indicator explanations
   - Market analysis assistance

## Technology Stack

- **Framework:** Streamlit
- **Language:** Python 3.11+
- **Data Source:** Yahoo Finance API (yfinance)
- **ML Models:** Prophet, Random Forest, LSTM
- **AI Integration:** OpenAI GPT-4, Google Gemini 2.0
- **Visualization:** Plotly, Matplotlib, Folium, Seaborn
- **Explainability:** SHAP, LIME

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Internet connection for real-time data

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/stock-market-analysis.git
cd stock-market-analysis
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API Keys (Optional)**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

5. **Run the application**
```bash
streamlit run src/streamlit_app.py --server.port 8507
```

6. **Access the application**
Open your browser and navigate to: `http://localhost:8507`

**Default Login:**
- Username: `demo`
- Password: `demo123`

## Project Structure

```
Project/
├── src/                          # Source code
│   ├── streamlit_app.py         # Main application
│   ├── data_loader.py           # Data fetching & processing
│   ├── technical_indicators.py  # Technical analysis
│   ├── prediction_models.py     # ML models
│   ├── explainable_ai.py       # XAI implementation
│   ├── chatbot_enhanced.py     # AI chatbot
│   ├── advanced_visualization.py # Advanced charts
│   └── auth_manager.py          # Authentication
├── data/                        # Data storage
├── config/                      # Configuration files
├── outputs/                     # Generated outputs
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Usage Guide

### Quick Start
1. Login with demo credentials
2. Select a stock symbol (e.g., AAPL, GOOGL, MSFT)
3. Choose analysis type from 6 tabs
4. Interact with charts and visualizations
5. Use AI chatbot for questions

### Technical Indicators
- **RSI > 70:** Overbought signal
- **RSI < 30:** Oversold signal
- **MACD Cross:** Buy/Sell signals
- **Bollinger Bands:** Volatility measure

## Performance

- **Load Time:** < 3 seconds
- **Chart Update:** < 2 seconds
- **AI Prediction:** 5-10 seconds
- **Chatbot Response:** 1-3 seconds
- **Concurrent Users:** 10-15 users

## Demo Mode

The application includes a demo mode with synthetic data for:
- Testing without internet connection
- Educational purposes
- Presentations and demonstrations

## Screenshots

*Add your screenshots here*

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is for educational purposes only. Not financial advice.

## Acknowledgments

- FPT University - DAP391m Course
- Yahoo Finance for market data
- OpenAI and Google for AI APIs
- Streamlit community

## Contact

**Student:** [Your Name]  
**Student ID:** [Your ID]  
**Email:** vudjeuvuj84@gmail.com  
**Course:** DAP391m - Data Analytics Project  
**University:** FPT University  
**Semester:** Fall 2025  

## Disclaimer

This application is developed for educational and research purposes. It does not provide financial advice. Users are responsible for their own investment decisions.

---

**Version:** 1.0  
**Last Updated:** November 3, 2025
