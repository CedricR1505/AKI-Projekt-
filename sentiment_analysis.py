"""
Sentiment-Analyse Modul für das Stock Dashboard
Enthält Funktionen für News-Abruf, Sentiment-Berechnung und Korrelationsanalyse
"""

import requests
import re
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from html import unescape

# ARIMA für Zeitreihen-Prognose
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    import warnings
    warnings.filterwarnings('ignore')
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

# VADER Sentiment Analyzer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    _analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    _analyzer = None


# ============== Konstanten ==============
PERIOD_DAYS_MAP = {
    "1d": 1, 
    "5d": 7, 
    "1mo": 30, 
    "3mo": 90, 
    "6mo": 180, 
    "1y": 365, 
    "5y": 1825
}

# Dynamische RSS-Feed Templates (mit Aktien-Symbol)
RSS_FEED_TEMPLATES = [
    # Google News - verschiedene Suchanfragen
    "https://news.google.com/rss/search?q={symbol}+stock&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=DE&ceid=DE:de",
    "https://news.google.com/rss/search?q={symbol}+shares&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+investor&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+earnings&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+quarterly&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+CEO&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+market&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+analysis&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+price&hl=en&gl=US&ceid=US:en",
    # Yahoo Finance RSS
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
]

# Statische RSS-Feeds (allgemeine Finanznachrichten)
STATIC_RSS_FEEDS = [
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories/", "name": "MarketWatch"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "name": "CNBC"},
    {"url": "https://feeds.marketwatch.com/marketwatch/marketpulse/", "name": "MarketWatch Pulse"},
    {"url": "https://www.investing.com/rss/news.rss", "name": "Investing.com"},
    {"url": "https://seekingalpha.com/market_currents.xml", "name": "Seeking Alpha"},
]


# ============== Hilfsfunktionen ==============
def get_cutoff_date(period: str):
    """Berechnet das Cutoff-Datum basierend auf dem gewählten Zeitraum."""
    days_back = PERIOD_DAYS_MAP.get(period, 30)
    return datetime.now() - timedelta(days=days_back)


def identify_source(feed_url: str) -> str:
    """Identifiziert die Nachrichtenquelle anhand der URL."""
    if "google.com" in feed_url:
        return "Google News"
    elif "yahoo.com" in feed_url:
        return "Yahoo Finance"
    elif "marketwatch.com" in feed_url:
        return "MarketWatch"
    elif "cnbc.com" in feed_url:
        return "CNBC"
    elif "investing.com" in feed_url:
        return "Investing.com"
    elif "seekingalpha.com" in feed_url:
        return "Seeking Alpha"
    elif "reuters" in feed_url:
        return "Reuters"
    return "Unbekannt"


def calculate_sentiment(text: str) -> float:
    """Berechnet den Sentiment-Score für einen Text."""
    if not VADER_AVAILABLE or _analyzer is None:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def parse_date(date_str: str):
    """Versucht verschiedene Datumsformate zu parsen. Gibt immer timezone-naive datetime zurück."""
    if not date_str:
        return datetime.now()
    
    date_formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%d %b %Y %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    result = None
    for fmt in date_formats:
        try:
            result = datetime.strptime(date_str.strip(), fmt)
            break
        except (ValueError, AttributeError):
            continue
    
    # Fallback mit pandas
    if result is None:
        try:
            result = pd.to_datetime(date_str).to_pydatetime()
        except:
            return datetime.now()
    
    # Timezone entfernen falls vorhanden (offset-naive machen)
    if result is not None and result.tzinfo is not None:
        result = result.replace(tzinfo=None)
    
    return result if result else datetime.now()


def fetch_rss_feed(url: str, timeout: int = 10):
    """Holt einen RSS-Feed und gibt die Items als Liste zurück."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        
        if resp.status_code != 200:
            return []
        
        content = resp.text
        
        # Items aus dem Feed extrahieren
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL | re.IGNORECASE)
        
        # Alternativ: entry-Tags (für Atom-Feeds)
        if not items:
            items = re.findall(r"<entry>(.*?)</entry>", content, re.DOTALL | re.IGNORECASE)
        
        return items
    except Exception as e:
        print(f"[RSS] Fehler bei {url}: {e}")
        return []


def parse_feed_item(item_xml: str, source_name: str) -> dict:
    """Parst ein RSS-Item und extrahiert die relevanten Daten."""
    # Titel extrahieren
    title_m = re.search(r"<title[^>]*>(.*?)</title>", item_xml, re.DOTALL | re.IGNORECASE)
    if title_m:
        title = title_m.group(1)
        # CDATA entfernen
        title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title, flags=re.DOTALL)
        title = unescape(title.strip())
        # HTML-Tags entfernen
        title = re.sub(r"<[^>]+>", "", title)
    else:
        title = ""
    
    # Datum extrahieren
    pub_m = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item_xml, re.DOTALL | re.IGNORECASE)
    if not pub_m:
        pub_m = re.search(r"<published[^>]*>(.*?)</published>", item_xml, re.DOTALL | re.IGNORECASE)
    if not pub_m:
        pub_m = re.search(r"<updated[^>]*>(.*?)</updated>", item_xml, re.DOTALL | re.IGNORECASE)
    
    pub_date = parse_date(pub_m.group(1) if pub_m else "")
    
    # Quelle aus dem Feed extrahieren (falls vorhanden)
    source_m = re.search(r"<source[^>]*>(.*?)</source>", item_xml, re.DOTALL | re.IGNORECASE)
    if source_m:
        original_source = unescape(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", source_m.group(1)).strip())
    else:
        original_source = source_name
    
    return {
        "title": title,
        "date": pub_date,
        "source": original_source
    }


def get_company_name(symbol: str) -> str:
    """Holt den Firmennamen von Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get("shortName", "") or info.get("longName", "")
        if name:
            # Ersten Teil des Namens verwenden (z.B. "Tesla" aus "Tesla, Inc.")
            return name.split(",")[0].split(" Inc")[0].split(" Corp")[0].strip()
    except:
        pass
    return ""


# ============== News-Abruf ==============
def fetch_news_from_feeds(symbol: str, period: str = "1mo", news_limit: int = 100):
    """
    Ruft News aus mehreren RSS-Feeds ab und berechnet Sentiment-Scores.
    Holt News aus ALLEN Feeds und begrenzt erst am Ende.
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        period: Zeitraum für die Analyse
        news_limit: Maximale Anzahl der News
        
    Returns:
        tuple: (news_items, sources_found)
    """
    cutoff_date = get_cutoff_date(period)
    
    all_news_items = []
    sources_count = {}
    seen_titles = set()
    
    # Firmenname für bessere Filterung holen
    company_name = get_company_name(symbol)
    search_terms = [symbol.upper()]
    if company_name:
        search_terms.append(company_name.upper())
    
    print(f"[Sentiment] Suche nach: {search_terms}")
    
    # 1. Dynamische Feeds (symbol-spezifisch) - alle abfragen
    for template in RSS_FEED_TEMPLATES:
        url = template.format(symbol=symbol)
        source_name = identify_source(url)
        
        items = fetch_rss_feed(url)
        print(f"[RSS] {source_name}: {len(items)} Items gefunden")
        
        for item_xml in items:
            parsed = parse_feed_item(item_xml, source_name)
            
            if not parsed["title"] or len(parsed["title"]) < 10:
                continue
            
            # Duplikate vermeiden
            title_hash = parsed["title"].lower()[:60]
            if title_hash in seen_titles:
                continue
            seen_titles.add(title_hash)
            
            # Zeitfilter
            try:
                item_date = parsed["date"]
                # Sicherstellen dass beide datetime naive sind
                if hasattr(item_date, 'tzinfo') and item_date.tzinfo is not None:
                    item_date = item_date.replace(tzinfo=None)
                if item_date < cutoff_date:
                    continue
            except Exception:
                pass
            
            # Sentiment berechnen
            score = calculate_sentiment(parsed["title"])
            
            # Quelle: Feed-Quelle (z.B. "Google News") + Original-Quelle falls vorhanden
            display_source = parsed["source"]
            if source_name == "Google News" and parsed["source"] != "Google News":
                display_source = f"{parsed['source']} (via Google)"
            
            all_news_items.append({
                "title": parsed["title"],
                "date": parsed["date"].strftime("%d.%m.%Y"),
                "date_obj": parsed["date"],
                "score": score,
                "source": display_source,
                "feed_source": source_name  # Für die Quellen-Zählung
            })
            
            sources_count[source_name] = sources_count.get(source_name, 0) + 1
    
    # 2. Statische Feeds (allgemeine News, nach Symbol/Firmenname filtern)
    for feed in STATIC_RSS_FEEDS:
        items = fetch_rss_feed(feed["url"])
        print(f"[RSS] {feed['name']}: {len(items)} Items gefunden")
        
        for item_xml in items:
            parsed = parse_feed_item(item_xml, feed["name"])
            
            if not parsed["title"]:
                continue
            
            # Prüfen ob Symbol oder Firmenname im Titel vorkommt
            title_upper = parsed["title"].upper()
            if not any(term in title_upper for term in search_terms):
                continue
            
            # Duplikate vermeiden
            title_hash = parsed["title"].lower()[:60]
            if title_hash in seen_titles:
                continue
            seen_titles.add(title_hash)
            
            # Zeitfilter
            try:
                item_date = parsed["date"]
                # Sicherstellen dass beide datetime naive sind
                if hasattr(item_date, 'tzinfo') and item_date.tzinfo is not None:
                    item_date = item_date.replace(tzinfo=None)
                if item_date < cutoff_date:
                    continue
            except Exception:
                pass
            
            score = calculate_sentiment(parsed["title"])
            
            all_news_items.append({
                "title": parsed["title"],
                "date": parsed["date"].strftime("%d.%m.%Y"),
                "date_obj": parsed["date"],
                "score": score,
                "source": feed["name"],
                "feed_source": feed["name"]  # Für die Quellen-Zählung
            })
            
            sources_count[feed["name"]] = sources_count.get(feed["name"], 0) + 1
    
    # Nach Datum sortieren (neueste zuerst) und auf Limit begrenzen
    all_news_items.sort(key=lambda x: x["date_obj"], reverse=True)
    news_items = all_news_items[:news_limit]
    
    # date_obj und feed_source entfernen für das Ergebnis
    final_sources = {}
    for item in news_items:
        # Quellen zählen basierend auf feed_source
        feed_src = item.get("feed_source", item["source"])
        final_sources[feed_src] = final_sources.get(feed_src, 0) + 1
        # Temporäre Felder entfernen
        if "date_obj" in item:
            del item["date_obj"]
        if "feed_source" in item:
            del item["feed_source"]
    
    # Quellen-Zusammenfassung erstellen
    sources_found = [f"{name} ({count})" for name, count in sorted(final_sources.items(), key=lambda x: -x[1])]
    
    print(f"[Sentiment] Insgesamt {len(news_items)} News (von {len(all_news_items)} gefunden) aus: {sources_found}")
    
    return news_items, sources_found


def fetch_news_for_correlation(symbol: str, period: str = "3mo"):
    """Holt News für die Korrelationsanalyse."""
    news_items, _ = fetch_news_from_feeds(symbol, period, 500)
    return news_items


# ============== Chart-Erstellung ==============
def create_sentiment_chart(symbol: str, hist, sentiment_daily) -> go.Figure:
    """Erstellt einen Dual-Axis Chart mit Kurs und Sentiment."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    is_positive = end_price >= start_price
    color_line = "#22c55e" if is_positive else "#ef4444"
    
    # Kurslinie
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines",
            name=f"{symbol} Kurs",
            line=dict(color=color_line, width=2),
            hovertemplate="%{y:.2f} USD<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Sentiment Balken
    if len(sentiment_daily) > 0:
        colors_bars = ["#22c55e" if s >= 0 else "#ef4444" for s in sentiment_daily.values]
        fig.add_trace(
            go.Bar(
                x=pd.to_datetime(sentiment_daily.index),
                y=sentiment_daily.values,
                name="Sentiment Score",
                marker_color=colors_bars,
                opacity=0.5,
                hovertemplate="Sentiment: %{y:.2f}<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Layout
    pct_change = ((end_price - start_price) / start_price) * 100
    sign = "+" if pct_change >= 0 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} Kurs vs. Nachrichten-Stimmung ({sign}{pct_change:.2f}%)",
            font=dict(size=16)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )
    
    fig.update_yaxes(title_text="Kurs (USD)", secondary_y=False, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1], gridcolor="#e5e7eb")
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    
    return fig


def create_correlation_chart(symbol: str, merged_df) -> go.Figure:
    """Erstellt einen Korrelations-Chart mit Kurs und Sentiment-Overlay."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol} Kurs mit Sentiment-Overlay", "Sentiment-Score (7-Tage Durchschnitt)")
    )
    
    start_price = merged_df["price"].iloc[0]
    end_price = merged_df["price"].iloc[-1]
    is_positive = end_price >= start_price
    color_price = "#22c55e" if is_positive else "#ef4444"
    
    # Kurs-Linie
    fig.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["price"],
            mode="lines",
            name="Kurs",
            line=dict(color=color_price, width=2),
            hovertemplate="%{y:.2f} USD<extra>Kurs</extra>"
        ),
        row=1, col=1
    )
    
    # Sentiment als Hintergrund-Färbung
    for i in range(len(merged_df) - 1):
        sentiment_val = merged_df["sentiment"].iloc[i]
        if abs(sentiment_val) > 0.1:
            fill_color = "rgba(34, 197, 94, 0.15)" if sentiment_val > 0 else "rgba(239, 68, 68, 0.15)"
            fig.add_vrect(
                x0=merged_df["date"].iloc[i],
                x1=merged_df["date"].iloc[i+1],
                fillcolor=fill_color,
                layer="below",
                line_width=0,
                row=1, col=1
            )
    
    # Sentiment-Chart unten
    colors_bars = ["#22c55e" if s > 0 else "#ef4444" for s in merged_df["sentiment_ma"]]
    fig.add_trace(
        go.Bar(
            x=merged_df["date"],
            y=merged_df["sentiment_ma"],
            name="Sentiment (MA7)",
            marker_color=colors_bars,
            opacity=0.7,
            hovertemplate="Sentiment: %{y:.3f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Nulllinie im Sentiment-Chart
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Layout
    fig.update_layout(
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_yaxes(title_text="Kurs (USD)", gridcolor="#e5e7eb", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", gridcolor="#e5e7eb", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    
    return fig


# ============== Analyse-Funktionen ==============
def analyze_sentiment(symbol: str, period: str = "1mo", news_limit: int = 100) -> dict:
    """
    Führt eine vollständige Sentiment-Analyse durch.
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        period: Zeitraum für die Analyse
        news_limit: Maximale Anzahl der News
        
    Returns:
        dict: Ergebnis mit allen Analysedaten oder Fehler
    """
    if not VADER_AVAILABLE:
        return {"error": "vaderSentiment nicht installiert. Installieren mit: pip install vaderSentiment"}
    
    symbol = symbol.strip().upper()
    
    try:
        # News abrufen
        news_items, sources_found = fetch_news_from_feeds(symbol, period, news_limit)
        
        if not news_items:
            return {"error": f"Keine News für '{symbol}' gefunden. Versuchen Sie einen längeren Zeitraum."}
        
        # DataFrame erstellen
        news_df = pd.DataFrame(news_items)
        news_df["date_parsed"] = pd.to_datetime(news_df["date"], format="%d.%m.%Y", errors="coerce")
        
        # Täglicher Durchschnitt
        sentiment_daily = news_df.groupby(news_df["date_parsed"].dt.date)["score"].mean()
        
        # Kursdaten abrufen
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period or "1mo")
        
        if hist.empty:
            return {"error": f"Keine Kursdaten für '{symbol}' verfügbar."}
        
        # Chart erstellen
        fig = create_sentiment_chart(symbol, hist, sentiment_daily)
        
        # Statistiken berechnen
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        avg_sentiment = news_df["score"].mean()
        
        return {
            "success": True,
            "symbol": symbol,
            "news_items": news_items,
            "sources_found": sources_found,
            "sentiment_daily": sentiment_daily,
            "figure": fig,
            "stats": {
                "avg_sentiment": avg_sentiment,
                "news_count": len(news_items),
                "sentiment_days": len(sentiment_daily),
                "start_price": start_price,
                "end_price": end_price,
                "pct_change": pct_change,
                "is_positive": end_price >= start_price,
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


def analyze_correlation(symbol: str, period: str = "3mo", news_limit: int = 500) -> dict:
    """
    Führt eine Korrelationsanalyse zwischen Kurs und Sentiment durch.
    
    Args:
        symbol: Aktiensymbol
        period: Zeitraum für die Analyse
        news_limit: Maximale Anzahl der News
        
    Returns:
        dict: Ergebnis mit Korrelationsdaten oder Fehler
    """
    if not VADER_AVAILABLE:
        return {"error": "vaderSentiment nicht installiert"}
    
    symbol = symbol.strip().upper()
    days_back = PERIOD_DAYS_MAP.get(period, 90)
    
    try:
        # News abrufen
        news_items, _ = fetch_news_from_feeds(symbol, period, news_limit)
        
        if len(news_items) < 5:
            return {"error": f"Zu wenige News für '{symbol}' gefunden ({len(news_items)} Artikel). Versuchen Sie einen längeren Zeitraum."}
        
        # DataFrame erstellen
        news_df = pd.DataFrame(news_items)
        news_df["date_parsed"] = pd.to_datetime(news_df["date"], format="%d.%m.%Y", errors="coerce")
        
        # Täglicher Durchschnitt
        sentiment_daily = news_df.groupby(news_df["date_parsed"].dt.date)["score"].mean().reset_index()
        sentiment_daily.columns = ["date", "sentiment"]
        sentiment_daily["date"] = pd.to_datetime(sentiment_daily["date"])
        
        # Kursdaten abrufen
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period or "3mo")
        
        if hist.empty:
            return {"error": f"Keine Kursdaten für '{symbol}' verfügbar."}
        
        # Kursdaten vorbereiten
        price_df = hist[["Close"]].reset_index()
        price_df.columns = ["date", "price"]
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)
        
        # Merge
        merged_df = pd.merge(price_df, sentiment_daily, on="date", how="left")
        merged_df["sentiment"] = merged_df["sentiment"].interpolate(method="linear").fillna(0)
        
        # Korrelation berechnen
        correlation = merged_df["price"].corr(merged_df["sentiment"])
        if pd.isna(correlation):
            correlation = 0.0
        
        # Rolling Average für Sentiment (7 Tage)
        merged_df["sentiment_ma"] = merged_df["sentiment"].rolling(window=7, min_periods=1).mean()
        
        # Chart erstellen
        fig = create_correlation_chart(symbol, merged_df)
        
        # Statistiken
        start_price = merged_df["price"].iloc[0]
        end_price = merged_df["price"].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        avg_sentiment = merged_df["sentiment"].mean()
        
        return {
            "success": True,
            "symbol": symbol,
            "correlation": correlation,
            "figure": fig,
            "merged_df": merged_df,
            "stats": {
                "news_count": len(news_items),
                "days_back": days_back,
                "start_price": start_price,
                "end_price": end_price,
                "pct_change": pct_change,
                "is_positive": end_price >= start_price,
                "avg_sentiment": avg_sentiment,
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_sentiment_label(score: float) -> tuple:
    """Gibt Label und Farbe für einen Sentiment-Score zurück."""
    if score > 0.05:
        return "positiv", "success"
    elif score < -0.05:
        return "negativ", "danger"
    return "neutral", "secondary"


def get_correlation_label(correlation: float) -> tuple:
    """Gibt Label und Farbe für einen Korrelationskoeffizienten zurück."""
    if correlation > 0.5:
        return "Stark positiv", "success"
    elif correlation > 0.3:
        return "Positiv", "success"
    elif correlation < -0.5:
        return "Stark negativ", "danger"
    elif correlation < -0.3:
        return "Negativ", "danger"
    return "Schwach/Neutral", "secondary"


# ============== ARIMA Prognose ==============
def analyze_forecast(symbol: str, history_period: str = "1y", forecast_days: int = 30) -> dict:
    """
    Führt eine ARIMA-basierte Kursprognose mit Trend-Korrektur durch.
    
    Für längere Prognosen wird ein Drift (historischer Trend) hinzugefügt,
    um Mean-Reversion zu vermeiden.
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        history_period: Historischer Zeitraum für das Training
        forecast_days: Anzahl der Tage für die Prognose
        
    Returns:
        dict: Ergebnis mit Prognosedaten oder Fehler
    """
    if not ARIMA_AVAILABLE:
        return {"error": "statsmodels nicht installiert. Installieren mit: pip install statsmodels"}
    
    symbol = symbol.strip().upper()
    
    try:
        # Kursdaten abrufen
        stock = yf.Ticker(symbol)
        hist = stock.history(period=history_period)
        
        if hist.empty or len(hist) < 30:
            return {"error": f"Nicht genügend Kursdaten für '{symbol}'. Mindestens 30 Datenpunkte benötigt."}
        
        # Daten vorbereiten - numerischen Index verwenden um Timestamp-Probleme zu vermeiden
        close_prices = hist["Close"].values
        dates_original = hist.index.tolist()
        
        # Konvertiere Daten zu Python datetime ohne Zeitzone
        dates_clean = []
        for d in dates_original:
            if hasattr(d, 'to_pydatetime'):
                dt = d.to_pydatetime()
            else:
                dt = pd.Timestamp(d).to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            dates_clean.append(dt)
        
        # DataFrame mit numerischem Index erstellen
        df = pd.DataFrame({
            "Close": close_prices,
            "Date": dates_clean
        })
        df = df.dropna()
        
        if len(df) < 30:
            return {"error": f"Nach Bereinigung nicht genügend Daten für '{symbol}'."}
        
        series = df["Close"].values
        
        # ========== Historischen Trend berechnen ==========
        # Log-Renditen für stabilen Drift
        log_returns = np.diff(np.log(series))
        daily_drift = np.mean(log_returns)  # Durchschnittliche tägliche Log-Rendite
        daily_volatility = np.std(log_returns)
        
        # Annualisierte Werte (für Info)
        annual_drift = daily_drift * 252
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # ========== ARIMA-Modell anpassen ==========
        # Stationarität prüfen
        d = 0
        try:
            adf_result = adfuller(series, autolag='AIC')
            if adf_result[1] > 0.05:
                d = 1
        except:
            d = 1
        
        # ARIMA-Modell mit Trend (trend='t' für linearen Trend bei d>0)
        best_aic = float('inf')
        best_order = (1, d, 1)
        best_model = None
        
        # Verschiedene ARIMA-Parameter testen
        for p in [1, 2, 3]:
            for q in [1, 2]:
                try:
                    # Bei d>0: trend='t' (linear) ist erlaubt, 'c' (konstant) nicht
                    # Bei d=0: trend='c' (konstant) ist erlaubt
                    trend_param = 't' if d > 0 else 'c'
                    model = ARIMA(series, order=(p, d, q), trend=trend_param)
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue
        
        if best_model is None:
            # Fallback: ARIMA ohne Trend-Term
            model = ARIMA(series, order=(1, 1, 1))
            best_model = model.fit()
            best_order = (1, 1, 1)
        
        # ========== Prognose mit Trend-Korrektur ==========
        # Basis-ARIMA Prognose
        try:
            arima_forecast = best_model.forecast(steps=forecast_days)
            if hasattr(arima_forecast, 'values'):
                arima_forecast = arima_forecast.values
        except:
            arima_forecast = best_model.predict(start=len(series), end=len(series) + forecast_days - 1)
            if hasattr(arima_forecast, 'values'):
                arima_forecast = arima_forecast.values
        
        # Für lange Prognosen: Trend-Drift hinzufügen, wenn ARIMA zur Mean-Reversion neigt
        current_price = series[-1]
        
        # Prüfen ob ARIMA zu stark zur Mean-Reversion neigt
        arima_end_change = (arima_forecast[-1] - current_price) / current_price
        expected_trend_change = daily_drift * forecast_days  # Erwartete Änderung basierend auf historischem Trend
        
        # Wenn der Unterschied zu groß ist (ARIMA ignoriert Trend), korrigieren
        if forecast_days > 90:  # Nur bei längeren Prognosen
            # Gewichteter Durchschnitt: Je länger die Prognose, desto mehr Gewicht auf den historischen Trend
            trend_weight = min(0.7, forecast_days / 1825)  # Max 70% Trend-Gewicht bei 5 Jahren
            
            # Trend-basierte Prognose mit GBM (exponentiellem Wachstum)
            trend_forecast = np.zeros(forecast_days)
            trend_forecast[0] = current_price
            for t in range(1, forecast_days):
                trend_forecast[t] = trend_forecast[t-1] * np.exp(daily_drift)
            
            # Kombiniere ARIMA und Trend
            forecast_mean = (1 - trend_weight) * arima_forecast + trend_weight * trend_forecast
        else:
            forecast_mean = arima_forecast
        
        # ========== Konfidenzintervall berechnen ==========
        residuals = best_model.resid
        std_err = np.std(residuals)
        
        # Konfidenzintervall wächst mit der Zeit (aber langsamer für lange Prognosen)
        # Verwende logarithmisches Wachstum statt Wurzel für stabilere Langzeitprognosen
        time_factor = np.log1p(np.arange(1, forecast_days + 1)) / np.log1p(30)  # Normalisiert auf 30 Tage
        ci_margin = 1.96 * std_err * (1 + time_factor * daily_volatility * np.sqrt(252))
        
        # Zusätzliche Unsicherheit für lange Prognosen
        if forecast_days > 365:
            long_term_uncertainty = (forecast_days / 365) * 0.1 * current_price  # 10% extra pro Jahr
            ci_margin = ci_margin + long_term_uncertainty * time_factor / time_factor[-1]
        
        forecast_ci_lower = forecast_mean - ci_margin
        forecast_ci_upper = forecast_mean + ci_margin
        
        # Sicherstellen, dass Preise nicht negativ werden
        forecast_ci_lower = np.maximum(forecast_ci_lower, current_price * 0.01)
        forecast_mean = np.maximum(forecast_mean, current_price * 0.05)
        
        forecast_ci = np.column_stack([forecast_ci_lower, forecast_ci_upper])
        
        # Prognose-Daten manuell berechnen (nur Werktage)
        last_date = dates_clean[-1]
        forecast_date_list = []
        current_date = last_date + timedelta(days=1)
        while len(forecast_date_list) < forecast_days:
            if current_date.weekday() < 5:  # Mo-Fr
                forecast_date_list.append(current_date)
            current_date = current_date + timedelta(days=1)
        
        # Chart erstellen
        fig = create_forecast_chart(symbol, df, forecast_date_list, forecast_mean, forecast_ci, best_order)
        
        # Statistiken berechnen
        current_price = df["Close"].iloc[-1]
        
        # forecast_mean kann numpy array oder pandas Series sein
        if hasattr(forecast_mean, 'iloc'):
            forecast_end_price = forecast_mean.iloc[-1]
        else:
            forecast_end_price = forecast_mean[-1]
            
        forecast_change = ((forecast_end_price - current_price) / current_price) * 100
        
        # Trend-Analyse
        forecast_trend = "steigend" if forecast_change > 2 else "fallend" if forecast_change < -2 else "seitwärts"
        
        # Konfidenzintervall am Ende
        if hasattr(forecast_ci, 'iloc'):
            ci_lower = forecast_ci.iloc[-1, 0]
            ci_upper = forecast_ci.iloc[-1, 1]
        else:
            ci_lower = forecast_ci[-1, 0]
            ci_upper = forecast_ci[-1, 1]
        
        return {
            "success": True,
            "symbol": symbol,
            "figure": fig,
            "stats": {
                "current_price": current_price,
                "forecast_price": forecast_end_price,
                "forecast_change": forecast_change,
                "forecast_trend": forecast_trend,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "forecast_days": forecast_days,
                "history_days": len(df),
                "arima_order": best_order,
                "aic": best_aic,
                "annual_drift": annual_drift * 100,  # In Prozent
                "annual_volatility": annual_volatility * 100,  # In Prozent
            }
        }
        
    except Exception as e:
        import traceback
        return {"error": f"Fehler bei der Prognose: {str(e)}\n{traceback.format_exc()}"}


def create_forecast_chart(symbol: str, hist_df, forecast_dates, forecast_mean, forecast_ci, arima_order) -> go.Figure:
    """Erstellt einen Chart mit historischen Daten und Prognose."""
    
    fig = go.Figure()
    
    # Historische Daten (letzte 90 Tage für bessere Übersicht)
    hist_display = hist_df.tail(90)
    
    # Verwende "Date" Spalte für x-Achse
    hist_dates = hist_display["Date"].tolist()
    hist_close = hist_display["Close"].tolist()
    
    start_price = hist_close[0]
    end_price = hist_close[-1]
    is_positive = end_price >= start_price
    color_hist = "#22c55e" if is_positive else "#ef4444"
    
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode="lines",
            name="Historischer Kurs",
            line=dict(color=color_hist, width=2),
            hovertemplate="%{y:.2f} USD<extra>Historisch</extra>"
        )
    )
    
    # Prognose-Linie
    forecast_values = list(forecast_mean) if hasattr(forecast_mean, '__iter__') else forecast_mean.tolist()
    color_forecast = "#3b82f6"  # Blau für Prognose
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines",
            name="Prognose",
            line=dict(color=color_forecast, width=2, dash="dash"),
            hovertemplate="%{y:.2f} USD<extra>Prognose</extra>"
        )
    )
    
    # Konfidenzintervall
    if hasattr(forecast_ci, 'iloc'):
        ci_upper = list(forecast_ci.iloc[:, 1])
        ci_lower = list(forecast_ci.iloc[:, 0])
    else:
        ci_upper = list(forecast_ci[:, 1])
        ci_lower = list(forecast_ci[:, 0])
        
    fig.add_trace(
        go.Scatter(
            x=list(forecast_dates) + list(reversed(forecast_dates)),
            y=ci_upper + list(reversed(ci_lower)),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Konfidenzintervall",
            hoverinfo="skip"
        )
    )
    
    # Verbindungslinie zwischen historisch und Prognose
    last_hist_date = hist_df["Date"].iloc[-1]
    # Konvertiere zu Python datetime für Plotly-Kompatibilität
    if hasattr(last_hist_date, 'to_pydatetime'):
        last_hist_date = last_hist_date.to_pydatetime()
    
    last_hist_price = hist_df["Close"].iloc[-1]
    first_forecast_price = forecast_values[0]
    
    fig.add_trace(
        go.Scatter(
            x=[last_hist_date, forecast_dates[0]],
            y=[last_hist_price, first_forecast_price],
            mode="lines",
            line=dict(color=color_forecast, width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    
    # Vertikale Linie am Prognosebeginn (ohne annotation_position wegen Plotly Bug)
    fig.add_shape(
        type="line",
        x0=last_hist_date,
        x1=last_hist_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", dash="dot", width=1)
    )
    fig.add_annotation(
        x=last_hist_date,
        y=1,
        yref="paper",
        text="Heute",
        showarrow=False,
        yanchor="bottom"
    )
    
    # Layout
    forecast_change = ((forecast_values[-1] - last_hist_price) / last_hist_price) * 100
    sign = "+" if forecast_change >= 0 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} Kursprognose - ARIMA{arima_order} ({sign}{forecast_change:.1f}% erwartet)",
            font=dict(size=16)
        ),
        xaxis_title="Datum",
        yaxis_title="Kurs (USD)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50),
        height=450
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    
    return fig


def get_forecast_label(change: float) -> tuple:
    """Gibt Label und Farbe für eine Prognose-Änderung zurück."""
    if change > 5:
        return "Stark steigend", "success"
    elif change > 2:
        return "Steigend", "success"
    elif change < -5:
        return "Stark fallend", "danger"
    elif change < -2:
        return "Fallend", "danger"
    return "Seitwärts", "secondary"


# ============== Monte-Carlo Simulation ==============
def analyze_monte_carlo(symbol: str, history_period: str = "1y", forecast_days: int = 30, num_simulations: int = 1000) -> dict:
    """
    Führt eine Monte-Carlo-Simulation für Kursprognosen durch.
    
    Die Simulation basiert auf der Geometric Brownian Motion (GBM):
    S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        history_period: Historischer Zeitraum für die Volatilitäts-Berechnung
        forecast_days: Anzahl der Tage für die Prognose
        num_simulations: Anzahl der Simulationen
        
    Returns:
        dict: Ergebnis mit Simulationsdaten oder Fehler
    """
    symbol = symbol.strip().upper()
    
    try:
        # Kursdaten abrufen
        stock = yf.Ticker(symbol)
        hist = stock.history(period=history_period)
        
        if hist.empty or len(hist) < 30:
            return {"error": f"Nicht genügend Kursdaten für '{symbol}'. Mindestens 30 Datenpunkte benötigt."}
        
        # Daten vorbereiten
        close_prices = hist["Close"].values
        dates_original = hist.index.tolist()
        
        # Konvertiere Daten zu Python datetime ohne Zeitzone
        dates_clean = []
        for d in dates_original:
            if hasattr(d, 'to_pydatetime'):
                dt = d.to_pydatetime()
            else:
                dt = pd.Timestamp(d).to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            dates_clean.append(dt)
        
        # DataFrame erstellen
        df = pd.DataFrame({
            "Close": close_prices,
            "Date": dates_clean
        })
        df = df.dropna()
        
        if len(df) < 30:
            return {"error": f"Nach Bereinigung nicht genügend Daten für '{symbol}'."}
        
        # Tägliche Renditen berechnen
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        
        # Parameter für GBM
        mu = returns.mean()  # Drift (durchschnittliche tägliche Rendite)
        sigma = returns.std()  # Volatilität
        
        # Aktueller Preis
        current_price = df["Close"].iloc[-1]
        
        # Monte-Carlo Simulation
        dt = 1  # 1 Tag
        
        # Simulationen durchführen
        np.random.seed(42)  # Für Reproduzierbarkeit
        simulations = np.zeros((num_simulations, forecast_days + 1))
        simulations[:, 0] = current_price
        
        for t in range(1, forecast_days + 1):
            # Zufällige Renditen aus Normalverteilung
            random_returns = np.random.normal(0, 1, num_simulations)
            # GBM Formel
            simulations[:, t] = simulations[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_returns
            )
        
        # Statistiken berechnen
        final_prices = simulations[:, -1]
        
        # Perzentile für verschiedene Konfidenzintervalle
        percentiles = {
            "p5": np.percentile(final_prices, 5),
            "p10": np.percentile(final_prices, 10),
            "p25": np.percentile(final_prices, 25),
            "p50": np.percentile(final_prices, 50),  # Median
            "p75": np.percentile(final_prices, 75),
            "p90": np.percentile(final_prices, 90),
            "p95": np.percentile(final_prices, 95),
        }
        
        mean_price = np.mean(final_prices)
        std_price = np.std(final_prices)
        
        # Wahrscheinlichkeit für verschiedene Szenarien
        prob_positive = np.sum(final_prices > current_price) / num_simulations * 100
        prob_up_10 = np.sum(final_prices > current_price * 1.10) / num_simulations * 100
        prob_down_10 = np.sum(final_prices < current_price * 0.90) / num_simulations * 100
        
        # Prognose-Daten (nur Werktage)
        last_date = dates_clean[-1]
        forecast_date_list = []
        current_date = last_date + timedelta(days=1)
        while len(forecast_date_list) <= forecast_days:
            if current_date.weekday() < 5:  # Mo-Fr
                forecast_date_list.append(current_date)
            current_date = current_date + timedelta(days=1)
        
        # Chart erstellen
        fig = create_monte_carlo_chart(
            symbol, df, forecast_date_list, simulations, 
            percentiles, mean_price, current_price
        )
        
        # Erwartete Änderung
        forecast_change = ((mean_price - current_price) / current_price) * 100
        
        return {
            "success": True,
            "symbol": symbol,
            "figure": fig,
            "simulations": simulations,
            "stats": {
                "current_price": current_price,
                "mean_price": mean_price,
                "median_price": percentiles["p50"],
                "std_price": std_price,
                "forecast_change": forecast_change,
                "forecast_days": forecast_days,
                "num_simulations": num_simulations,
                "history_days": len(df),
                "mu": mu * 252,  # Annualisierte Drift
                "sigma": sigma * np.sqrt(252),  # Annualisierte Volatilität
                "percentiles": percentiles,
                "prob_positive": prob_positive,
                "prob_up_10": prob_up_10,
                "prob_down_10": prob_down_10,
            }
        }
        
    except Exception as e:
        import traceback
        return {"error": f"Fehler bei der Monte-Carlo-Simulation: {str(e)}\n{traceback.format_exc()}"}


def create_monte_carlo_chart(symbol: str, hist_df, forecast_dates, simulations, 
                             percentiles, mean_price, current_price) -> go.Figure:
    """Erstellt einen Chart mit historischen Daten und Monte-Carlo-Simulation."""
    
    fig = go.Figure()
    
    # Historische Daten (letzte 90 Tage für bessere Übersicht)
    hist_display = hist_df.tail(90)
    hist_dates = hist_display["Date"].tolist()
    hist_close = hist_display["Close"].tolist()
    
    start_price = hist_close[0]
    end_price = hist_close[-1]
    is_positive = end_price >= start_price
    color_hist = "#22c55e" if is_positive else "#ef4444"
    
    # Historischer Kurs
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode="lines",
            name="Historischer Kurs",
            line=dict(color=color_hist, width=2),
            hovertemplate="%{y:.2f} USD<extra>Historisch</extra>"
        )
    )
    
    # Einige Simulationspfade anzeigen (max. 100 für Performance)
    num_display = min(100, simulations.shape[0])
    for i in range(num_display):
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=simulations[i, :],
                mode="lines",
                line=dict(color="rgba(100, 100, 100, 0.05)", width=0.5),
                showlegend=False,
                hoverinfo="skip"
            )
        )
    
    # Perzentil-Bänder (90% Konfidenzintervall)
    p5_values = np.percentile(simulations, 5, axis=0)
    p95_values = np.percentile(simulations, 95, axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=list(forecast_dates) + list(reversed(forecast_dates)),
            y=list(p95_values) + list(reversed(p5_values)),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Konfidenzintervall",
            hoverinfo="skip"
        )
    )
    
    # Perzentil-Bänder (50% Konfidenzintervall)
    p25_values = np.percentile(simulations, 25, axis=0)
    p75_values = np.percentile(simulations, 75, axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=list(forecast_dates) + list(reversed(forecast_dates)),
            y=list(p75_values) + list(reversed(p25_values)),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="50% Konfidenzintervall",
            hoverinfo="skip"
        )
    )
    
    # Median-Linie
    median_values = np.percentile(simulations, 50, axis=0)
    color_forecast = "#3b82f6"  # Blau für Prognose
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=median_values,
            mode="lines",
            name="Median (50%)",
            line=dict(color=color_forecast, width=3),
            hovertemplate="%{y:.2f} USD<extra>Median</extra>"
        )
    )
    
    # Mittelwert-Linie
    mean_values = np.mean(simulations, axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=mean_values,
            mode="lines",
            name="Mittelwert",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            hovertemplate="%{y:.2f} USD<extra>Mittelwert</extra>"
        )
    )
    
    # Verbindungslinie zwischen historisch und Prognose
    last_hist_date = hist_df["Date"].iloc[-1]
    if hasattr(last_hist_date, 'to_pydatetime'):
        last_hist_date = last_hist_date.to_pydatetime()
    
    last_hist_price = hist_df["Close"].iloc[-1]
    
    fig.add_trace(
        go.Scatter(
            x=[last_hist_date, forecast_dates[0]],
            y=[last_hist_price, simulations[0, 0]],
            mode="lines",
            line=dict(color=color_forecast, width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    
    # Vertikale Linie am Prognosebeginn
    fig.add_shape(
        type="line",
        x0=last_hist_date,
        x1=last_hist_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", dash="dot", width=1)
    )
    fig.add_annotation(
        x=last_hist_date,
        y=1,
        yref="paper",
        text="Heute",
        showarrow=False,
        yanchor="bottom"
    )
    
    # Layout
    forecast_change = ((mean_price - current_price) / current_price) * 100
    sign = "+" if forecast_change >= 0 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} Monte-Carlo Simulation ({sign}{forecast_change:.1f}% erwartet)",
            font=dict(size=16)
        ),
        xaxis_title="Datum",
        yaxis_title="Kurs (USD)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    
    return fig


def get_monte_carlo_label(prob_positive: float) -> tuple:
    """Gibt Label und Farbe für Monte-Carlo-Wahrscheinlichkeit zurück."""
    if prob_positive > 70:
        return "Sehr bullisch", "success"
    elif prob_positive > 55:
        return "Bullisch", "success"
    elif prob_positive < 30:
        return "Sehr bearisch", "danger"
    elif prob_positive < 45:
        return "Bearisch", "danger"
    return "Neutral", "secondary"
