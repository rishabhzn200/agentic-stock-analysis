import logging
from pathlib import Path

from .fetch_data import get_stock_data
from .features import compute_features
from .model import load_model, train_model, predict, MODEL_PATH

logger = logging.getLogger(__name__)

FEATURES = ["RSI", "EMA_10", "EMA_50", "MACD"]


def predict_stock(ticker):
    """
    Fetch data, compute features, load (or train) model, and predict
    whether the ticker goes up (1) or down (0) tomorrow.

    Returns:
        pred (int): 1 for UP, 0 for DOWN
        latest_features (dict): the latest row's feature values
    """
    # Use longer period for more training data
    logger.info(f"Running prediction for {ticker}...")
    df = get_stock_data(ticker, period="2y")
    df = compute_features(df)

    # Try to load an existing model; if not found, train a new one
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        logger.info("Loading existing model...")
        model = load_model()
    else:
        logger.info("Training new model...")
        model = train_model(df, FEATURES)

    latest = df[FEATURES].iloc[[-1]]
    pred = predict(model, latest)[0]
    latest_dict = latest.to_dict(orient="records")[0]
    logger.info(f"Prediction complete for {ticker}.")
    return pred, latest_dict
