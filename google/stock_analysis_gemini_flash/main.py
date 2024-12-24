import os
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import yfinance as yf

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

# Sidebar configuration
with st.sidebar:
    st.title("ℹ️ Configuration")

    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()

    st.info(
        "This tool provides AI-powered stock market analysis using advanced "
        "predictive models and market data analysis."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for informational purposes only and "
        "does not constitute financial advice. Consult with a licensed financial advisor before making investment decisions."
    )

# Configure stock analysis agent
stock_agent = Agent(
    model=Gemini(
        api_key=st.session_state.GOOGLE_API_KEY,
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGo()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

if not stock_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Stock Market Analysis Query Template
query = """
You are a highly skilled financial analyst specializing in stock market trends and predictive analysis. Analyze the provided stock data and structure your response as follows:

### 1. Overview of the Stock
- Provide an overview of the stock (ticker, company name, sector, etc.)
- Summarize recent performance and trends (1-week, 1-month, and 6-month views)
- Highlight key technical and fundamental indicators

### 2. Prediction Analysis
- Forecast the stock price for the next 1 week, 1 month, and 6 months
- Identify key drivers behind the predictions (e.g., earnings reports, market conditions)
- Provide confidence intervals for each prediction

### 3. Risk Assessment
- Outline potential risks associated with the stock
- Discuss market volatility and external factors (e.g., interest rates, geopolitical issues)
- Provide a risk rating (Low/Medium/High)

### 4. Investment Advice
- Offer actionable advice for investors based on the analysis
- Specify short-term and long-term strategies
- Highlight entry/exit points if applicable

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent news articles or reports about the stock or industry
- Search for analyst recommendations and ratings
- Provide links to at least 2-3 relevant sources

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

# Main Application Layout
st.title("📈 Stock Market Prediction Agent")
st.write("Enter a company name to fetch stock data and analyze it.")

# Create containers for better organization
input_container = st.container()
data_container = st.container()
analysis_container = st.container()

with input_container:
    company_name = st.text_input(
        "Enter Company Name:",
        help="Provide the name of the company to fetch stock data."
    )
    fetch_button = st.button(
        "🔍 Fetch and Analyze Stock Data",
        type="primary",
        use_container_width=True
    )

if company_name and fetch_button:
    with data_container:
        try:
            # Fetch stock data using yfinance
            ticker = yf.Ticker(company_name)
            if 'symbol' not in ticker.info:
                raise ValueError("Invalid ticker symbol or company name.")
            
            ticker_symbol = ticker.info['symbol']
            stock_data = yf.download(ticker_symbol, period="6mo")

            if stock_data.empty:
                raise ValueError("No stock data found for the given company.")

            st.write(f"### Stock Data for {company_name}", stock_data.head())

            # Save data to a temporary file for analysis
            temp_data_path = "temp_stock_data.csv"
            stock_data.to_csv(temp_data_path)

            with analysis_container:
                with st.spinner("🔄 Analyzing stock data... Please wait."):
                    try:
                        response = stock_agent.run(query, files=[temp_data_path])
                        
                        # Display analysis results
                        if response and response.content:
                            st.markdown("### 📋 Analysis Results")
                            st.markdown("---")
                            st.markdown(response.content)
                        else:
                            st.warning("No analysis results were returned.")
                    except Exception as analysis_error:
                        st.error(f"Analysis error: {analysis_error}")
                    finally:
                        if os.path.exists(temp_data_path):
                            os.remove(temp_data_path)
        except Exception as fetch_error:
            st.error(f"Error fetching stock data: {fetch_error}")
