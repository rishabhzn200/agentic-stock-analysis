import argparse
import logging
import sys
import yfinance as yf

from stock_predictor.log_config import setup_logging
from stock_predictor.predict_stock import predict_stock
from stock_predictor.ai_explainer import explain_trend

setup_logging()
logger = logging.getLogger(__name__)


def is_valid_ticker(ticker):
    """
    Validate the ticker symbol using yfinance by checking
    if there is any recent historical data.
    """
    try:
        # Try to fetch 1 month of data
        data = yf.Ticker(ticker).history(period="1mo")
        return data is not None and not data.empty
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stock prediction for a given ticker symbol."
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol, e.g. AAPL, MSFT, TSLA",
    )
    args = parser.parse_args()
    return args.ticker.strip().upper()


def main():
    ticker = parse_args()

    logger.info(f"Validating ticker: {ticker}")
    if not is_valid_ticker(ticker):
        logger.error(
            f"Ticker '{ticker}' is invalid or has no recent data. "
            "Please check the symbol and try again."
        )
        sys.exit(1)

    logger.info(f"Starting prediction for {ticker}...")
    try:
        pred, indicators = predict_stock(ticker)
    except Exception as e:
        logger.error(f"Failed to run prediction for {ticker}: {e}")
        sys.exit(1)

    print("[RUN] Prediction complete.")
    logger.info(f"Prediction: {'UP' if pred == 1 else 'DOWN'}")
    logger.info(f"Indicators: {indicators}")

    # AI explanation from OpenAI
    try:
        explanation = explain_trend(ticker, pred, indicators)
        logger.info("AI Explanation:")
        logger.info(explanation)
    except Exception as e:
        logger.error(f"AI explanation skipped: {e}")


if __name__ == "__main__":
    main()
