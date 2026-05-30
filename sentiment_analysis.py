# src/sentiment_analysis.py
# Run in Spyder → df, sentiment visible
# 3 sources: finnhub (recent), alphavantage (historical), yahoo (live)

import argparse
import yaml
import pandas as pd
import logging
import os
import torch
import requests
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import finnhub
from datetime import datetime, timedelta
import numpy as np

# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Configuration loaded: {cfg}")
    return cfg

# --------------------------------------------------------------------- #
# 2. Finnhub – Recent news (last 30 days)
# --------------------------------------------------------------------- #
def setup_finnhub_client():
    load_dotenv()
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY missing in .env")
    return finnhub.Client(api_key=key)

def fetch_finnhub_news(client, symbol, start, end):
    """Fetch recent company news (free tier: ~30 days)."""
    today = datetime.now().strftime("%Y-%m-%d")
    if end > today:
        end = today
    try:
        raw = client.company_news(symbol, _from=start, to=end)
        items = []
        for entry in raw:
            if isinstance(entry, dict) and "headline" in entry:
                try:
                    date = pd.to_datetime(entry["datetime"], unit="s").floor("D")
                    items.append({"date": date, "headline": entry["headline"]})
                except:
                    continue
        logger.info(f"Finnhub: {len(items)} recent headlines")
        return items
    except Exception as e:
        logger.error(f"Finnhub error: {e}")
        return []

# --------------------------------------------------------------------- #
# 3. Alpha Vantage – Historical news (2023-2024)
# --------------------------------------------------------------------- #
def fetch_alphavantage_news(symbol, start_dt, end_dt):
    """Fetch historical news from Alpha Vantage."""
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        logger.warning("ALPHAVANTAGE_API_KEY missing")
        return []

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "time_from": start_dt.strftime("%Y%m%dT0000"),
        "time_to": end_dt.strftime("%Y%m%dT2359"),
        "limit": 200,
        "apikey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "feed" not in data:
            logger.warning(f"Alpha Vantage: {data.get('Note', 'No data')}")
            return []
        items = []
        for item in data["feed"]:
            try:
                date = pd.to_datetime(item["time_published"][:8], format="%Y%m%d").floor("D")
                items.append({"date": date, "headline": item["title"]})
            except:
                continue
        logger.info(f"Alpha Vantage: {len(items)} historical headlines")
        return items
    except Exception as e:
        logger.error(f"Alpha Vantage error: {e}")
        return []

# --------------------------------------------------------------------- #
# 4. Yahoo Finance – Live news (today only)
# --------------------------------------------------------------------- #
def fetch_yahoo_news():
    """Fetch live market news from Yahoo RSS (today only)."""
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    try:
        feed = feedparser.parse(url)
        today = datetime.now().date()
        items = []
        for entry in feed.entries:
            if not hasattr(entry, "published_parsed"):
                continue
            try:
                pub_date = pd.to_datetime(datetime(*entry.published_parsed[:6])).floor("D")
                if pub_date == today:
                    items.append({"date": today, "headline": entry.title})
            except:
                continue
        logger.info(f"Yahoo RSS: {len(items)} live headlines")
        return items
    except Exception as e:
        logger.error(f"Yahoo RSS error: {e}")
        return []

# --------------------------------------------------------------------- #
# 5. FinBERT – Sentiment
# --------------------------------------------------------------------- #
logger.info("Initializing FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()
logger.info(f"FinBERT successfully loaded on {device}")

def compute_finbert_sentiment(texts):
    """Computes sentiment scores using the globally loaded FinBERT model."""
    def score(t):
        if not t or t.strip() == "": 
            return np.nan  # Usamos NaN en lugar de 0.0 para días sin noticias
            
        try:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            
            pos, neg, neu = probs[1].item(), probs[0].item(), probs[2].item()
            return pos if pos > max(neg, neu) else -neg if neg > max(pos, neu) else neu * 0.5
        except Exception as e:
            logger.error(f"Error scoring text: {e}")
            return np.nan
            
    return [score(t) for t in texts]
# --------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

symbol = cfg["stock_symbol"]
start_str = cfg["start_date"]
end_str = cfg["end_date"]

raw_dir = cfg["raw_dir"]
raw_csv = f"{raw_dir}/{cfg['stock_symbol']}_raw.csv"
source = cfg.get("sentiment_source", "finnhub").lower()

processed_dir = cfg["processed_dir"]
out_csv = f"{processed_dir}/{cfg['stock_symbol']}_sentiment_{source}.csv"

start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()

df = pd.read_csv(raw_csv)
df["Date"] = pd.to_datetime(df["Date"]).dt.floor("D").dt.normalize()
df = df.sort_values("Date").reset_index(drop=True)
logger.info(f"Loaded {len(df)} trading days")

# --- Fetch news (COMBINED SOURCES) ---
news_items = []

# AlphaVantage (historical)
news_items += fetch_alphavantage_news(symbol, start_dt, end_dt)

# Finnhub (recent)
try:
    client = setup_finnhub_client()
    news_items += fetch_finnhub_news(client, symbol, start_str, end_str)
except Exception as e:
    logger.warning(f"Finnhub failed: {e}")

# Yahoo (live / same-day)
news_items += fetch_yahoo_news()

# --- Map to dates (CORREGIDO) ---
news_df = pd.DataFrame(news_items)
news_df["date"] = pd.to_datetime(news_df["date"]).dt.floor("D")
news_df["date"] = news_df["date"].dt.normalize()
df["Date"] = pd.to_datetime(df["Date"]).dt.floor("D")

news_grouped = news_df.groupby("date")["headline"].apply(list).to_dict()
print("TOTAL news_items:", len(news_items))
print("SAMPLE:", news_items[:3])

window = 7  
texts = []

for date in df["Date"]:
    relevant = []

    for i in range(window + 1):
        past = date - pd.Timedelta(days=i)

        headlines = news_grouped.get(past, [])
        
        # decay temporal (más reciente = más peso)
        weight = 1 / (i + 1)

        relevant.extend([h for h in headlines[:5]])
        # cap por día
    text = " | ".join(relevant[:5])
    texts.append(text)
texts = [
    t if t.strip() != "" else None
    for t in texts
]

df["news"] = texts
df["sentiment_raw"] = compute_finbert_sentiment(texts)


df["sentiment_raw"] = pd.to_numeric(df["sentiment_raw"], errors="coerce")

# no inventes ceros → usa forward fill
df["sentiment"] = df["sentiment_raw"].ffill()

# smoothing real de señal
df["sentiment_smooth"] = (
    df["sentiment"]
    .rolling(10, min_periods=1)
    .mean()
)
# Métricas limpias para el Log
active_days = df["sentiment_raw"].dropna()
mean_active = active_days.mean() if not active_days.empty else 0.0

logger.info(f"Total trading days: {len(df)}")
logger.info(f"Trading days with news: {len(active_days)}")
logger.info(f"Real Sentiment mean (only days with news): {mean_active:.4f}")
logger.info(f"Global Sentiment mean (including zeros): {df['sentiment'].mean():.4f}")

# Guardar resultados
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
logger.info(f"Saved → {out_csv}")

print("unique news dates:", len(news_grouped))
print("trading dates:", len(df["Date"]))
print("matches:", len(set(df["Date"]) & set(news_grouped.keys())))