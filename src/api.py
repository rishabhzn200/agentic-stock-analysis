import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_predictor.agent_graph import build_agent_graph
from stock_predictor.log_config import setup_logging
from stock_predictor.predict_stock import predict_stock
from stock_predictor.ai_explainer import explain_trend

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

AGENT_GRAPH = build_agent_graph()

app = FastAPI(
    title="Stock Predictor API",
    description="Predicts next day stock movement and gets an AI explanation",
    version="1.0",
)


class AnalyzeRequest(BaseModel):
    ticker: str
    explain: bool = True


class AnalyzeResponse(BaseModel):
    ticker: str
    model_prediction: str
    indicators: dict
    explanation: str | None


class AgentAnalyzeRequest(BaseModel):
    ticker: str
    question: str


class AgentAnalyzeResponse(BaseModel):
    ticker: str
    question: str
    model_prediction: str | None = None
    news_sentiment_label: str | None = None
    news_sentiment_score: float | None = None
    alignment: str | None = None
    news_headlines_used: list[str] | None = None
    report: str


@app.get("/health_check")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    ticker = request.ticker.strip().upper()
    logger.info(f"/analyze called for ticker={ticker}, explain={request.explain}")

    # Run the prediction pipeline
    try:
        pred, indicators = predict_stock(ticker)
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    prediction_str = "UP" if pred == 1 else "DOWN"

    # AI explanation
    explanation = None
    if request.explain:
        try:
            explanation = explain_trend(ticker, pred, indicators)
        except Exception as e:
            # logger.error(f"Explanation failed for {ticker}: {e}")
            logger.exception(f"Explanation failed for {ticker}: {e}")
            # Do not fail the whole request if explanation breaks
            explanation = None

    return AnalyzeResponse(
        ticker=ticker,
        model_prediction=prediction_str,
        indicators=indicators,
        explanation=explanation,
    )


@app.post("/analyze_agent", response_model=AgentAnalyzeResponse)
def analyze_agent(request: AgentAnalyzeRequest):
    ticker = request.ticker.strip().upper()
    question = request.question.strip()

    logger.info(f"/analyze_agent called ticker={ticker}")

    try:
        final_state = AGENT_GRAPH.invoke({"ticker": ticker, "question": question})
    except Exception as e:
        logger.exception(f"Agent graph failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return AgentAnalyzeResponse(
        ticker=ticker,
        question=question,
        model_prediction=final_state.get("prediction"),
        news_sentiment_label=final_state.get("news_sentiment_label"),
        news_sentiment_score=final_state.get("news_sentiment_score"),
        alignment=final_state.get("alignment"),
        news_headlines_used=final_state.get("news_headlines_used"),
        report=final_state.get("report", ""),
    )
