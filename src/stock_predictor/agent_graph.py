import json
import logging
from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field

import requests
import yfinance as yf
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from .config import get_news_config
from .predict_stock import predict_stock

logger = logging.getLogger(__name__)

ALLOWED_NEWS_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "fool.com",
    "oilprice.com",
    "economictimes.indiatimes.com",
]


class AgentState(TypedDict, total=False):
    ticker: str
    ticker_metadata: Dict[str, Any]
    question: str
    prediction: str
    indicators: str
    news_sentiment_label: str
    news_sentiment_score: float
    alignment: str
    news_headlines_used: List[str]
    news_search_terms: List[str]
    news_provider: str
    news_items: str
    report: str
    error: str


class NewsQuery(BaseModel):
    terms: List[str] = Field(
        ..., description="Search terms for querying financial news APIs."
    )


class NewsSentiment(BaseModel):
    label: str = Field(
        ..., description="One of POSITIVE, NEGATIVE, NEUTRAL, MIXED, NO_NEWS"
    )
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score in [-1, 1]")


def _fetch_news_from_news_api(
    query: str, limit: int = 5, domains: list[str] | None = None
) -> List[Dict[str, Any]]:
    config = get_news_config()
    if not config.newsapi_key:
        raise ValueError("NEWSAPI_API_KEY is not set")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": limit,
        "sortBy": "publishedAt",
        "searchIn": "title,description",
        "apiKey": config.newsapi_key,
    }

    if domains:
        params["domains"] = ",".join(domains)

    response = requests.get(url, params=params, timeout=10)
    logger.info(f"[agent] fetch news response={response}")
    response.raise_for_status()
    data = response.json()

    items: List[Dict[str, Any]] = []
    for article in data.get("articles", [])[:limit]:
        items.append(
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source": (article.get("source") or {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
            }
        )
    return items


def _fetch_news_from_yfinance(ticker_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    yf_ticker = yf.Ticker(ticker_name)
    yf_news = getattr(yf_ticker, "news", []) or []
    yf_news = yf_news[:limit]

    items: List[Dict[str, Any]] = []
    for news in yf_news:
        items.append(
            {
                "title": news.get("title"),
                "description": None,
                "source": news.get("publisher"),
                "url": news.get("link"),
                "published_at": news.get("providerPublishTime"),
            }
        )
    return items


def fetch_ticker_metadata_node(state: AgentState) -> AgentState:
    ticker_name = state.get("ticker")
    if not ticker_name:
        raise ValueError("Missing 'ticker' in agent state")

    ticker = yf.Ticker(ticker_name)
    info = ticker.info or {}

    metadata = {
        "symbol": ticker_name,
        "shortName": info.get("shortName"),
        "longName": info.get("longName"),
        "quoteType": info.get("quoteType"),
        "category": info.get("category"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
    }

    logger.info(f"[agent] metadata_node ticker={ticker_name} metadata={metadata}")
    return {"ticker_metadata": metadata}


def plan_news_query_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    if not ticker:
        raise ValueError("Missing 'ticker' in agent state")

    ticker_metadata = state.get("ticker_metadata", {}) or {}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(NewsQuery)
    prompt = f"""
    Generate precise search terms for a News API query, using provided ticker metadata.

    Ticker: {ticker}
    Metadata (JSON):
    {json.dumps(ticker_metadata, indent=2)}

    Return JSON with:
    {{"terms": ["..."]}}

    Rules:
    - Prefer official names from metadata (shortName/longName) over generic terms.
    - Include 3 to 6 terms max.
    - Avoid overly generic terms like just "ETF" or "fund" or "stock".
    - For ETFs/commodities, include 1-2 asset-specific phrases if clearly implied (e.g., for gold ETF include "gold price").
    - Only include the raw ticker symbol if it is likely unambiguous (>=4 chars) OR if metadata strongly indicates it is commonly referenced.
    """.strip()

    result: NewsQuery = structured_llm.invoke(prompt)

    logger.info(f"[news_query] news_query_node response: {result}")

    terms = [term.strip() for term in result.terms if term and term.strip()]

    # If model returns nothing, fallback to metadata names or ticker
    if not terms:
        fallback = [
            ticker_metadata.get("shortName"),
            ticker_metadata.get("longName"),
            ticker,
        ]
        terms = [x for x in fallback if x]

    logger.info(
        f"[agent] plan_news_query_node completed: ticker={ticker}, terms={terms}"
    )

    return {"news_search_terms": terms}


def predict_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    logger.info(f"[agent] predict_node ticker={ticker}")
    pred_int, indicators = predict_stock(ticker)
    prediction = "UP" if pred_int == 1 else "DOWN"

    return {
        "prediction": prediction,
        "indicators": indicators,
    }


def news_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    config = get_news_config()
    provider = config.provider
    terms = state.get("news_search_terms") or [state["ticker"]]
    query = " OR ".join(f'"{t}"' for t in terms if t)

    logger.info(f"[agent] news_node ticker={ticker} provider={provider}")

    items: List[Dict[str, Any]]
    try:
        if provider == "newsapi":
            items = _fetch_news_from_news_api(
                query, limit=5, domains=ALLOWED_NEWS_DOMAINS
            )

            # if domain is too restrictive
            if not items:
                logger.info(
                    "[agent] No results from trusted domains, retrying without domain restriction"
                )
                items = _fetch_news_from_news_api(query, limit=5)
        else:
            items = _fetch_news_from_yfinance(ticker, limit=5)
    except Exception as e:
        logger.exception(
            f"[agent] news fetch failed provider={provider}, falling back to yfinance: {e}"
        )
        provider = "yfinance"
        items = _fetch_news_from_yfinance(ticker, limit=5)

    logger.info(
        f"[agent] news_node completed: provider={provider}, items_count={len(items)}, Items={items}"
    )

    return {
        "news_provider": provider,
        "news_items": items,
    }


def news_sentiment_node(state: AgentState) -> AgentState:
    headlines = [
        news.get("title")
        for news in (state.get("news_items") or [])
        if news.get("title")
    ][:5]

    if not headlines:
        return {
            "news_sentiment_label": "NO_NEWS",
            "news_sentiment_score": 0.0,
            "news_headlines_used": [],
        }

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured = llm.with_structured_output(NewsSentiment)

    prompt = f"""
    You are scoring SHORT-TERM news tone for the next 1-3 trading days for the given asset.
    Use ONLY the headlines; do not invent details.

    Headlines:
    {json.dumps(headlines, indent=2)}

    Return:
        - label: POSITIVE, NEGATIVE, NEUTRAL, MIXED
        - score: float in [-1,1] where -1 is very negative, +1 very positive, 0 neutral
    """.strip()

    result: NewsSentiment = structured.invoke(prompt)

    label = result.label.strip().upper()
    if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}:
        label = "NEUTRAL"

    return {
        "news_sentiment_label": label,
        "news_sentiment_score": float(result.score),
        "news_headlines_used": headlines,
    }


def alignment_node(state: AgentState) -> AgentState:
    pred = (state.get("prediction") or "").upper()
    label = (state.get("news_sentiment_label") or "").upper()

    # If no usable news
    if label in {"NO_NEWS"} or not pred:
        return {"alignment": "UNKNOWN"}

    # Map sentiment label to direction
    if label == "POSITIVE":
        news_dir = "UP"
    elif label == "NEGATIVE":
        news_dir = "DOWN"
    else:
        return {"alignment": "UNKNOWN"}  # NEUTRAL / MIXED can't confidently map

    alignment = "ALIGNED" if news_dir == pred else "CONFLICT"
    return {"alignment": alignment}


def summarize_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    question = state["question"]
    prediction = state.get("prediction")
    indicators = state.get("indicators", {})
    provider = state.get("news_provider", "unknown")
    news_items = state.get("news_items", [])
    news_sent_label = state.get("news_sentiment_label", "NO_NEWS")
    news_sent_score = state.get("news_sentiment_score", 0.0)
    alignment = state.get("alignment", "UNKNOWN")

    logger.info(f"[agent] summarize_node ticker={ticker}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    # Keep prompt structured and compact (avoid token bloat)
    news_compact = [
        {
            "title": x.get("title"),
            "source": x.get("source"),
            "published_at": x.get("published_at"),
        }
        for x in news_items
        if x.get("title")
    ]

    logger.info(
        f"[sumarize] summarize_node starting: provider={provider}, news_compact={news_compact}"
    )

    prompt = f"""
    You are an AI assistant that summarizes short-term stock signals for educational purposes (not financial advice).

    Ticker: {ticker}
    User question: {question}

    Model prediction horizon: next trading day (directional UP/DOWN).
    Model prediction for tomorrow: {prediction}

    News sentiment horizon: short-term tone for the next 1–3 trading days (may not match next-day move).
    News sentiment: {news_sent_label} (score={news_sent_score})
    Model vs news alignment: {alignment}

    Latest indicators (JSON):
    {json.dumps(indicators, indent=2)}

    Recent news headlines (provider={provider}) (JSON):
    {json.dumps(news_compact, indent=2)}

    Write a concise report with:
    1) One-line summary (include model prediction + whether news aligns/conflicts)
    2) Indicators interpretation (RSI/EMA/MACD) in simple terms
    3) News context:
    - Use ONLY the headlines provided (no hallucinations).
    - If headlines are weakly related to the ticker, say that and avoid causal claims.
    4) What to watch next day (2-3 bullet points)
    5) A short "Headlines used" list (max 5 titles)

    Include a short disclaimer that this is not financial advice.
    """.strip()

    result = llm.invoke(prompt)
    report = result.content if hasattr(result, "content") else str(result)

    return {"report": report}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("metadata", fetch_ticker_metadata_node)
    graph.add_node("plan_news_query", plan_news_query_node)
    graph.add_node("news", news_node)
    graph.add_node("news_sentiment", news_sentiment_node)
    graph.add_node("predict", predict_node)
    graph.add_node("alignment", alignment_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "metadata")
    graph.add_edge("metadata", "plan_news_query")
    graph.add_edge("plan_news_query", "news")
    graph.add_edge("news", "news_sentiment")
    graph.add_edge("news_sentiment", "predict")
    graph.add_edge("predict", "alignment")
    graph.add_edge("alignment", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
