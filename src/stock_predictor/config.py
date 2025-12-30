import os

from dataclasses import dataclass


@dataclass
class NewsConfig:
    provider: str
    newsapi_key: str


def get_news_config() -> NewsConfig:
    provider = os.getenv("NEWS_PROVIDER", "yfinance").lower()
    api_key = os.getenv("NEWSAPI_API_KEY")
    return NewsConfig(provider=provider, newsapi_key=api_key)
