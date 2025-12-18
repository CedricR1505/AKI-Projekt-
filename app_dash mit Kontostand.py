import dash
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import json
import re
from pathlib import Path
from datetime import datetime
from html import unescape
import feedparser
import pandas as pd
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# ============== Daten-Pfade ==============
DATA_DIR = Path(__file__).parent / "gui"
DATA_DIR.mkdir(exist_ok=True)
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRANSACTIONS_FILE = DATA_DIR / "transactions.json"
BALANCE_FILE = DATA_DIR / "balance.json"

# ============== Market Overview Symbole ==============
MARKET_OVERVIEW_SYMBOLS = [
    {"name": "DAX", "symbol": "^GDAXI", "decimals": 0},
    {"name": "MDAX", "symbol": "^MDAXI", "decimals": 0},
    {"name": "SDAX", "symbol": "^SDAXI", "decimals": 0},
    {"name": "Dow", "symbol": "^DJI", "decimals": 0},
    {"name": "Nasdaq", "symbol": "^IXIC", "decimals": 0},
    {"name": "Gold", "symbol": "GC=F", "decimals": 2},
    {"name": "Brent", "symbol": "BZ=F", "decimals": 2},
    {"name": "BTC", "symbol": "BTC-USD", "decimals": 0},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "decimals": 4, "invert": True},
]

# ============== Hilfsfunktionen ==============
def load_portfolio():
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        except:
            return []
    return []

def save_portfolio(data):
    PORTFOLIO_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_transactions():
    if TRANSACTIONS_FILE.exists():
        try:
            return json.loads(TRANSACTIONS_FILE.read_text(encoding="utf-8"))
        except:
            return []
    return []

def save_transaction(tx):
    txs = load_transactions()
    txs.append(tx)
    TRANSACTIONS_FILE.write_text(json.dumps(txs, indent=2), encoding="utf-8")

def load_balance():
    if BALANCE_FILE.exists():
        try:
            return float(json.loads(BALANCE_FILE.read_text(encoding="utf-8")))
        except:
            return 10000.0
    return 10000.0

def save_balance(balance):
    BALANCE_FILE.write_text(json.dumps(balance), encoding="utf-8")

def fetch_price(symbol):
    try:
        t = yf.Ticker(symbol)
        fast = getattr(t, "fast_info", None)
        if fast:
            price = getattr(fast, "last_price", None)
            prev = getattr(fast, "previous_close", None)
            return price, prev
    except:
        pass
    return None, None

def fetch_name(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return info.get("longName") or info.get("shortName") or symbol
    except:
        return symbol

def fetch_stock_history(symbol, period="1mo", interval="1d"):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval=interval)
        return hist
    except:
        return None

def search_stocks(query):
    if not query or len(query) < 2:
        return []
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
        results = []
        for q in data.get("quotes", []):
            if q.get("quoteType") in ["EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "CURRENCY"]:
                results.append({
                    "symbol": q.get("symbol"),
                    "name": q.get("shortname") or q.get("longname") or q.get("symbol"),
                    "exchange": q.get("exchange", "")
                })
        return results
    except:
        return []

def fetch_google_news(symbol, limit=20):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=de&gl=DE&ceid=DE:de"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
        news = []
        for item in items[:limit]:
            title_m = re.search(r"<title>(.*?)</title>", item)
            link_m = re.search(r"<link>(.*?)</link>", item)
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item)
            source_m = re.search(r"<source.*?>(.*?)</source>", item)
            title = unescape(title_m.group(1)) if title_m else "News"
            link = link_m.group(1) if link_m else ""
            pub = pub_m.group(1) if pub_m else ""
            source = unescape(source_m.group(1)) if source_m else ""
            news.append({"title": title, "link": link, "pubDate": pub, "source": source, "symbol": symbol})
        return news
    except:
        return []

def format_volume(vol):
    if vol is None:
        return "n/a"
    if vol >= 1_000_000_000:
        return f"{vol/1_000_000_000:.2f}B"
    if vol >= 1_000_000:
        return f"{vol/1_000_000:.2f}M"
    if vol >= 1_000:
        return f"{vol/1_000:.1f}K"
    return str(vol)

def create_stock_chart(symbol, period="1mo", interval="1d"):
    hist = fetch_stock_history(symbol, period, interval)
    if hist is None or hist.empty:
        fig = go.Figure()
        fig.add_annotation(text="Keine Daten verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    
    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    is_positive = end_price >= start_price
    color = "#22c55e" if is_positive else "#ef4444"
    
    # Y-Achse zoomen
    y_min = hist["Close"].min()
    y_max = hist["Close"].max()
    y_range = y_max - y_min
    if y_range < 0.01 * y_max:
        padding = 0.005 * y_max
    else:
        padding = y_range * 0.1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({34 if is_positive else 239}, {197 if is_positive else 68}, {94 if is_positive else 68}, 0.1)",
        name=symbol,
        hovertemplate="%{y:.2f}<extra></extra>"
    ))
    
    pct_change = ((end_price - start_price) / start_price) * 100
    sign = "+" if pct_change >= 0 else ""
    
    fig.update_layout(
        title=dict(text=f"{symbol} ({sign}{pct_change:.2f}%)", font=dict(size=16, color=color)),
        yaxis=dict(range=[y_min - padding, y_max + padding], tickformat=",.2f", gridcolor="#e5e7eb"),
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode="x unified",
        showlegend=False
    )
    return fig

def create_portfolio_pie_chart(portfolio):
    if not portfolio:
        fig = go.Figure()
        fig.add_annotation(text="Portfolio ist leer", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    
    labels = []
    values = []
    colors = []
    
    for item in portfolio:
        symbol = item["symbol"]
        qty = item["qty"]
        current_price, _ = fetch_price(symbol)
        if current_price:
            value = qty * current_price
            values.append(value)
            labels.append(symbol)
            # Zufällige Farben oder feste Palette
            colors.append(f"hsl({hash(symbol) % 360}, 70%, 50%)")
    
    if not values:
        fig = go.Figure()
        fig.add_annotation(text="Keine aktuellen Preise verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    
    fig.update_layout(
        title="Portfolio-Zusammensetzung",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_portfolio_value_chart(portfolio):
    if not portfolio:
        fig = go.Figure()
        fig.add_annotation(text="Portfolio ist leer", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    
    symbols = []
    values = []
    invested = []
    pnl = []
    
    for item in portfolio:
        symbol = item["symbol"]
        qty = item["qty"]
        buy_price = item.get("buy_price") or item.get("avg_price", 0)
        inv = qty * buy_price
        invested.append(inv)
        
        current_price, _ = fetch_price(symbol)
        if current_price:
            val = qty * current_price
            values.append(val)
            pnl_val = val - inv
            pnl.append(pnl_val)
            symbols.append(fetch_name(symbol))
        else:
            values.append(inv)
            pnl.append(0)
            symbols.append(fetch_name(symbol))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=symbols, y=invested, name='Investiert', marker_color='blue'))
    fig.add_trace(go.Bar(x=symbols, y=values, name='Aktueller Wert', marker_color='green'))
    fig.add_trace(go.Bar(x=symbols, y=pnl, name='Gewinn/Verlust', marker_color='red' if sum(pnl) < 0 else 'green'))
    
    fig.update_layout(
        title="Portfolio-Übersicht pro Position",
        xaxis_title="Symbol",
        yaxis_title="Wert (USD)",
        barmode='group',
        legend_title="Legende"
    )
    
    return fig


def create_portfolio_total_value_chart(portfolio, days=30):
    """Erstellt eine Zeitreihe des Gesamtwerts des Portfolios über die letzten `days` Tage.
    Für jede Position werden historische Schlusskurse abgefragt und mit der Menge multipliziert.
    """
    import datetime

    if not portfolio:
        fig = go.Figure()
        fig.add_annotation(text="Keine Positionen im Portfolio", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    totals = {}
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)

    for item in portfolio:
        symbol = item.get("symbol")
        qty = item.get("qty", 0)
        if not symbol or qty == 0:
            continue

        # hole historische Preise (täglicher Schlusskurs)
        hist = fetch_stock_history(symbol, period=f"{days}d", interval="1d")
        if hist is None or hist.empty:
            continue

        for idx, row in hist.iterrows():
            # idx ist ein Timestamp
            try:
                dt = idx.date()
            except Exception:
                # falls idx kein DatetimeIndex-Element ist
                dt = datetime.date.fromtimestamp(idx.timestamp())

            if dt < start or dt > end:
                continue

            close = row.get("Close") if isinstance(row, dict) or hasattr(row, "get") else row["Close"]
            try:
                value = float(close) * qty
            except Exception:
                continue

            totals.setdefault(dt, 0.0)
            totals[dt] += value

    if not totals:
        fig = go.Figure()
        fig.add_annotation(text="Keine historischen Preise verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    dates = sorted(totals.keys())
    values = [totals[d] for d in dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name="Gesamtwert", line=dict(color="#0ea5a4")))

    # Gesamtinvestition als Referenz (aktueller investierter Betrag)
    total_invested = sum((item.get("qty", 0) * (item.get("buy_price") or item.get("avg_price", 0))) for item in portfolio)
    fig.add_trace(go.Scatter(x=dates, y=[total_invested] * len(dates), mode="lines", name="Investiert (aktuell)", line=dict(color="#2563eb", dash="dash")))

    fig.update_layout(
        title="Gesamtwert des Portfolios (letzte %s Tage)" % days,
        xaxis_title="Datum",
        yaxis_title="Wert (USD)",
        hovermode="x unified",
        legend_title="Legende",
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig

# ============== App Initialisierung ==============
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Stock Dashboard"

# ============== Layout ==============
def create_market_ticker():
    return dbc.Row([
        dbc.Col(html.Div(id=f"ticker-{s['name']}", className="text-center p-2", 
                         style={"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa"}), 
                width="auto") 
        for s in MARKET_OVERVIEW_SYMBOLS
    ], className="g-2 p-2 bg-light mb-3", justify="center")

app.layout = dbc.Container([
    dcc.Interval(id="market-interval", interval=15000, n_intervals=0),
    dcc.Store(id="selected-ticker", data=None),
    dcc.Store(id="portfolio-store", data=load_portfolio()),
    dcc.Store(id="search-results-store", data=[]),
    
    # Market Ticker Bar
    html.H4("📈 Stock Dashboard", className="text-center my-3"),
    create_market_ticker(),
    
    # Tabs
    dbc.Tabs([
        # Portfolio Tab
        dbc.Tab(label="Portfolio", children=[
            dbc.Tabs([
                # Übersicht Sub-Tab
                dbc.Tab(label="Übersicht", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("💰 Buy/Sell", id="btn-buy-sell", color="primary", size="sm"),
                                dbc.Button("📋 Transactions", id="btn-transactions", color="secondary", size="sm"),
                                dbc.Button("💵 Kontostand", id="btn-kontostand", color="info", size="sm"),
                            ], className="mb-3"),
                        ], width=12),
                    ]),
                    html.Div(id="portfolio-table"),
                    html.Div(id="portfolio-summary", className="mt-3"),
                    dbc.Row([
                        dbc.Col([dcc.Graph(id="portfolio-chart", style={"height": "400px"})], width=12),
                    ], className="mt-3"),
                ], className="p-3"),
                
                # Wertentwicklung Sub-Tab
                dbc.Tab(label="Wertentwicklung", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H6("📈 Portfolio-Wertentwicklung"),
                            html.P("Zeigt die Entwicklung des Portfolio-Werts über Zeit, inklusive Gewinne und Verluste.", className="text-muted"),
                            dcc.Graph(id="portfolio-value-chart", style={"height": "360px"}),
                            html.Hr(),
                            html.H6("📊 Gesamtwert des Portfolios"),
                            dcc.Graph(id="portfolio-total-value-chart", style={"height": "300px"}),
                        ], width=12),
                    ]),
                ], className="p-3"),
            ]),
        ], className="p-3"),
        
        # Aktien Tab
        dbc.Tab(label="Aktien", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Input(id="stock-search", placeholder="Aktie suchen (z.B. Apple, TSLA)...", type="text", debounce=True),
                ], width=6),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("1T", id="btn-1d", size="sm", outline=True, color="primary"),
                        dbc.Button("1W", id="btn-1w", size="sm", outline=True, color="primary"),
                        dbc.Button("1M", id="btn-1m", size="sm", outline=True, color="primary", active=True),
                        dbc.Button("3M", id="btn-3m", size="sm", outline=True, color="primary"),
                        dbc.Button("1J", id="btn-1y", size="sm", outline=True, color="primary"),
                        dbc.Button("Max", id="btn-max", size="sm", outline=True, color="primary"),
                    ]),
                ], width=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([dcc.Graph(id="stock-chart", style={"height": "400px"})], width=8),
                dbc.Col([
                    html.H6("📰 News"),
                    html.Div(id="stock-news", style={"maxHeight": "380px", "overflowY": "auto"})
                ], width=4),
            ]),
        ], className="p-3"),
        
        # News Tab
        dbc.Tab(label="News", children=[
            dbc.Button("🔄 Aktualisieren", id="btn-refresh-news", color="light", size="sm", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div(id="market-news", style={"maxHeight": "600px", "overflowY": "auto"})
                ], width=12),
            ]),
        ], className="p-3"),
        
        # AI Analysis Tab
        dbc.Tab(label="AI Analysis", children=[
            dbc.Tabs([
                # Sentiment-Analyse Sub-Tab
                dbc.Tab(label="Sentiment-Analyse", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H6("📊 Sentiment-Analyse"),
                            html.P("Analysiere die Stimmung zu einer Aktie basierend auf aktuellen Nachrichten (Google News) und vergleiche mit dem Kursverlauf.", className="text-muted"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("🔍"),
                                dbc.Input(id="sentiment-search-input", placeholder="Aktie suchen (z.B. Apple, Tesla, Microsoft)...", type="text", debounce=True),
                            ], className="mb-2"),
                            dbc.Select(
                                id="sentiment-stock-dropdown",
                                options=[],
                                placeholder="Bitte zuerst eine Aktie suchen...",
                                className="mb-3"
                            ),
                        ], width=5),
                        dbc.Col([
                            html.Small("Zeitraum", className="text-muted"),
                            dbc.Select(
                                id="sentiment-period-select",
                                options=[
                                    {"label": "1 Tag", "value": "1d"},
                                    {"label": "1 Woche", "value": "5d"},
                                    {"label": "1 Monat", "value": "1mo"},
                                    {"label": "3 Monate", "value": "3mo"},
                                    {"label": "6 Monate", "value": "6mo"},
                                    {"label": "1 Jahr", "value": "1y"},
                                    {"label": "5 Jahre", "value": "5y"},
                                ],
                                value="1mo",
                                className="mb-3"
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Small("News-Anzahl", className="text-muted"),
                            dbc.Select(
                                id="sentiment-news-count",
                                options=[
                                    {"label": "50 News", "value": "50"},
                                    {"label": "100 News", "value": "100"},
                                    {"label": "200 News", "value": "200"},
                                    {"label": "500 News", "value": "500"},
                                    {"label": "1000 News", "value": "1000"},
                                ],
                                value="100",
                                className="mb-3"
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Small(" ", className="d-block"),
                            dbc.Button("🔍 Analysieren", id="btn-sentiment-analyze", color="primary", className="w-100"),
                        ], width=2),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="sentiment-loading",
                                type="circle",
                                children=[
                                    html.Div(id="ai-sentiment-output", className="mt-3"),
                                ]
                            ),
                        ], width=12),
                    ]),
                ], className="p-3"),
                
                # Prognose Sub-Tab
                dbc.Tab(label="Prognose", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H6("🔮 Prognose"),
                            html.P("Erhalte KI-gestützte Prognosen für Aktienkurse oder Markttrends.", className="text-muted"),
                            dbc.Textarea(id="ai-forecast-input", placeholder="Geben Sie eine Aktie oder einen Markt für die Prognose ein...", className="mb-3"),
                            dbc.Button("🔍 Prognostizieren", id="btn-forecast-analyze", color="success", className="mb-3"),
                            html.Div(id="ai-forecast-output", className="mt-3"),
                        ], width=12),
                    ]),
                ], className="p-3"),
                
                # Korrelation Sub-Tab
                dbc.Tab(label="Korrelation", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H6("📈 Kurs-Sentiment Korrelation"),
                            html.P("Analysiere die Korrelation zwischen Aktienkurs und Nachrichten-Sentiment über Zeit.", className="text-muted"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("🔍"),
                                dbc.Input(id="corr-search-input", placeholder="Aktie suchen (z.B. Apple, Tesla)...", type="text", debounce=True),
                            ], className="mb-2"),
                            dbc.Select(
                                id="corr-stock-dropdown",
                                options=[],
                                placeholder="Bitte zuerst eine Aktie suchen...",
                                className="mb-3"
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Small("Zeitraum", className="text-muted"),
                            dbc.Select(
                                id="corr-period-select",
                                options=[
                                    {"label": "1 Woche", "value": "5d"},
                                    {"label": "1 Monat", "value": "1mo"},
                                    {"label": "3 Monate", "value": "3mo"},
                                    {"label": "6 Monate", "value": "6mo"},
                                    {"label": "1 Jahr", "value": "1y"},
                                    {"label": "5 Jahre", "value": "5y"},
                                ],
                                value="3mo",
                                className="mb-3"
                            ),
                        ], width=2),
                        dbc.Col([
                            html.Small(" ", className="d-block"),
                            dbc.Button("📊 Korrelation berechnen", id="btn-corr-analyze", color="info", className="w-100"),
                        ], width=3),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="corr-loading",
                                type="circle",
                                children=[
                                    html.Div(id="corr-output", className="mt-3"),
                                ]
                            ),
                        ], width=12),
                    ]),
                ], className="p-3"),
            ]),
        ], className="p-3"),
    ]),
    
    # Buy/Sell Modal
    dbc.Modal([
        dbc.ModalHeader("💰 Kaufen / Verkaufen"),
        dbc.ModalBody([
            dbc.Input(id="buy-search", placeholder="Aktie suchen...", className="mb-2", debounce=True),
            html.Div(id="buy-search-results", style={"maxHeight": "150px", "overflowY": "auto"}),
            html.Hr(),
            html.Div(id="buy-stock-info"),
            dcc.Graph(id="buy-chart", style={"height": "250px"}),
            dbc.Row([
                dbc.Col([dbc.Label("Anzahl:"), dbc.Input(id="buy-qty", type="number", value=1, min=1)], width=6),
                dbc.Col([
                    html.Div(id="buy-total", className="mt-4"),
                    html.Div(id="buy-balance", className="text-muted small mt-2")
                ], width=6),
            ], className="mt-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("✅ Kaufen", id="btn-confirm-buy", color="success"),
            dbc.Button("❌ Verkaufen", id="btn-confirm-sell", color="danger"),
            dbc.Button("Schließen", id="btn-close-modal", color="secondary"),
        ]),
    ], id="buy-sell-modal", size="lg"),
    
    # Transactions Modal
    dbc.Modal([
        dbc.ModalHeader("📋 Transaktionen"),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([dbc.Select(id="tx-year", options=[{"label": "Alle Jahre", "value": "all"}])], width=4),
                dbc.Col([dbc.Select(id="tx-month", options=[{"label": "Alle Monate", "value": "all"}] + 
                                    [{"label": m, "value": str(i)} for i, m in enumerate(
                                        ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"], 1)])], width=4),
                dbc.Col([dbc.Select(id="tx-type", options=[
                    {"label": "Alle", "value": "all"},
                    {"label": "Käufe", "value": "buy"},
                    {"label": "Verkäufe", "value": "sell"}
                ])], width=4),
            ], className="mb-3"),
            html.Div(id="transactions-table"),
            html.Div(id="transactions-summary", className="mt-3"),
        ]),
        dbc.ModalFooter(dbc.Button("Schließen", id="btn-close-tx", color="secondary")),
    ], id="transactions-modal", size="xl"),
    
    # Kontostand Modal
    dbc.Modal([
        dbc.ModalHeader("💵 Kontostand"),
        dbc.ModalBody([
            html.H5("Aktueller Kontostand"),
            html.Div(id="kontostand-display", className="mt-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([dbc.Label("Betrag (USD):"), dbc.Input(id="balance-amount", type="number", value=100, min=0, step=0.01)], width=6),
                dbc.Col([
                    dbc.Button("➕ Einzahlung", id="btn-deposit", color="success", className="me-2"),
                    dbc.Button("➖ Auszahlung", id="btn-withdraw", color="danger"),
                ], width=6, className="d-flex align-items-end"),
            ], className="mt-3"),
            html.Div(id="balance-message", className="mt-3"),
        ]),
        dbc.ModalFooter(dbc.Button("Schließen", id="btn-close-kontostand", color="secondary")),
    ], id="kontostand-modal"),
    
    # Ticker Detail Modal
    dbc.Modal([
        dbc.ModalHeader(id="ticker-modal-header"),
        dbc.ModalBody([
            html.Div(id="ticker-modal-stats", className="mb-3"),
            dbc.ButtonGroup([
                dbc.Button("1T", id="ticker-btn-1d", size="sm", outline=True, color="primary", active=True),
                dbc.Button("1W", id="ticker-btn-1w", size="sm", outline=True, color="primary"),
                dbc.Button("1M", id="ticker-btn-1m", size="sm", outline=True, color="primary"),
                dbc.Button("3M", id="ticker-btn-3m", size="sm", outline=True, color="primary"),
            ], className="mb-3"),
            dcc.Graph(id="ticker-modal-chart", style={"height": "400px"}),
        ]),
        dbc.ModalFooter(dbc.Button("Schließen", id="btn-close-ticker", color="secondary")),
    ], id="ticker-modal", size="lg"),
    
    # Hidden Store für aktuellen Ticker im Modal
    dcc.Store(id="current-ticker-symbol", data=None),
    
], fluid=True)

# ============== Callbacks ==============

# Market Ticker Update
@callback(
    [Output(f"ticker-{s['name']}", "children") for s in MARKET_OVERVIEW_SYMBOLS] +
    [Output(f"ticker-{s['name']}", "style") for s in MARKET_OVERVIEW_SYMBOLS],
    Input("market-interval", "n_intervals")
)
def update_market_tickers(n):
    texts = []
    styles = []
    for s in MARKET_OVERVIEW_SYMBOLS:
        price, prev = fetch_price(s["symbol"])
        if s.get("invert") and price:
            price = 1 / price
            if prev:
                prev = 1 / prev
        
        if price is None:
            texts.append(html.Span([html.B(s["name"]), ": n/a"]))
            styles.append({"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa", "padding": "8px"})
        else:
            decimals = s.get("decimals", 2)
            formatted = f"{price:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            if prev:
                diff = price - prev
                color = "#22c55e" if diff > 0.0001 else "#ef4444" if diff < -0.0001 else "#000000"
            else:
                color = "#000000"
            
            texts.append(html.Span([html.B(s["name"]), f": {formatted}"], style={"color": color, "fontWeight": "bold"}))
            styles.append({"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa", "padding": "8px"})
    
    return texts + styles

# Stock Search & Chart
@callback(
    Output("stock-chart", "figure"),
    Output("stock-news", "children"),
    Input("stock-search", "value"),
    Input("btn-1d", "n_clicks"),
    Input("btn-1w", "n_clicks"),
    Input("btn-1m", "n_clicks"),
    Input("btn-3m", "n_clicks"),
    Input("btn-1y", "n_clicks"),
    Input("btn-max", "n_clicks"),
    prevent_initial_call=True
)
def update_stock_view(search, n1d, n1w, n1m, n3m, n1y, nmax):
    triggered = ctx.triggered_id
    
    period_map = {
        "btn-1d": ("1d", "5m"),
        "btn-1w": ("5d", "15m"),
        "btn-1m": ("1mo", "1d"),
        "btn-3m": ("3mo", "1d"),
        "btn-1y": ("1y", "1wk"),
        "btn-max": ("max", "1mo"),
    }
    
    period, interval = period_map.get(triggered, ("1mo", "1d"))
    
    if not search or len(search) < 2:
        return go.Figure(), html.P("Bitte Aktie suchen...")
    
    results = search_stocks(search)
    if not results:
        return go.Figure(), html.P("Keine Ergebnisse")
    
    symbol = results[0]["symbol"]
    fig = create_stock_chart(symbol, period, interval)
    
    news = fetch_google_news(symbol, 10)
    news_items = [
        dbc.Card([
            dbc.CardBody([
                html.A(n["title"], href=n["link"], target="_blank", className="text-decoration-none"),
                html.Small(f" — {n['source']}", className="text-muted d-block")
            ], className="p-2")
        ], className="mb-2") for n in news
    ] if news else [html.P("Keine News gefunden")]
    
    return fig, news_items

# Portfolio Display
@callback(
    Output("portfolio-table", "children"),
    Output("portfolio-summary", "children"),
    Output("portfolio-chart", "figure"),
    Output("portfolio-value-chart", "figure"),
    Output("portfolio-total-value-chart", "figure"),
    Input("portfolio-store", "data")
)
def update_portfolio(portfolio):
    if not portfolio:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Portfolio ist leer", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        empty_fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return html.P("Portfolio ist leer. Nutze Buy/Sell um Aktien hinzuzufügen.", className="text-muted"), "", empty_fig, empty_fig, empty_fig
    
    rows = []
    total_invested = 0
    total_value = 0
    
    for item in portfolio:
        symbol = item["symbol"]
        name = fetch_name(symbol)
        qty = item["qty"]
        buy_price = item.get("buy_price") or item.get("avg_price", 0)
        invested = qty * buy_price
        total_invested += invested
        
        current_price, _ = fetch_price(symbol)
        if current_price:
            value = qty * current_price
            total_value += value
            pnl = value - invested
            pnl_pct = (pnl / invested) * 100 if invested else 0
            
            rows.append({
                "Symbol": symbol,
                "Name": name,
                "Anzahl": qty,
                "Kaufkurs": f"{buy_price:.2f}",
                "Aktuell": f"{current_price:.2f}",
                "Investiert": f"{invested:.2f}",
                "Wert": f"{value:.2f}",
                "P/L": f"{pnl:+.2f} ({pnl_pct:+.2f}%)"
            })
        else:
            rows.append({
                "Symbol": symbol,
                "Name": name,
                "Anzahl": qty,
                "Kaufkurs": f"{buy_price:.2f}",
                "Aktuell": "n/a",
                "Investiert": f"{invested:.2f}",
                "Wert": "n/a",
                "P/L": "n/a"
            })
    
    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["Symbol", "Name", "Anzahl", "Kaufkurs", "Aktuell", "Investiert", "Wert", "P/L"]],
        style_cell={"textAlign": "center", "padding": "10px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": "{P/L} contains '+'", "column_id": "P/L"}, "color": "#22c55e", "fontWeight": "bold"},
            {"if": {"filter_query": "{P/L} contains '-'", "column_id": "P/L"}, "color": "#ef4444", "fontWeight": "bold"},
        ]
    )
    
    total_pnl = total_value - total_invested
    summary = dbc.Alert([
        html.B("Gesamt: "),
        f"Investiert: {total_invested:,.2f} USD | Wert: {total_value:,.2f} USD | ",
        html.Span(f"P/L: {total_pnl:+,.2f} USD", style={"color": "#22c55e" if total_pnl >= 0 else "#ef4444", "fontWeight": "bold"})
    ], color="light")
    
    chart = create_portfolio_pie_chart(portfolio)
    value_chart = create_portfolio_value_chart(portfolio)
    total_chart = create_portfolio_total_value_chart(portfolio)
    
    return table, summary, chart, value_chart, total_chart

# Market News
@callback(
    Output("market-news", "children"),
    Input("btn-refresh-news", "n_clicks"),
    prevent_initial_call=False
)
def update_market_news(n):
    all_news = []
    for target in ["DAX", "Nasdaq", "S&P 500", "Bitcoin", "Gold"]:
        news = fetch_google_news(target, 5)
        all_news.extend(news)
    
    if not all_news:
        return html.P("Keine News verfügbar")
    
    return [
        dbc.Card([
            dbc.CardBody([
                html.H6(html.A(n["title"], href=n["link"], target="_blank", className="text-decoration-none")),
                html.Small([
                    html.Span(n.get("source", ""), className="text-muted"),
                    " | ",
                    html.Span(n.get("symbol", ""), className="badge bg-secondary")
                ])
            ], className="p-2")
        ], className="mb-2") for n in all_news
    ]

# Buy/Sell Modal Toggle
@callback(
    Output("buy-sell-modal", "is_open"),
    Input("btn-buy-sell", "n_clicks"),
    Input("btn-close-modal", "n_clicks"),
    Input("btn-confirm-buy", "n_clicks"),
    Input("btn-confirm-sell", "n_clicks"),
    State("buy-sell-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_buy_sell_modal(n1, n2, n3, n4, is_open):
    return not is_open

# Kontostand Modal Toggle
@callback(
    Output("kontostand-modal", "is_open"),
    Input("btn-kontostand", "n_clicks"),
    Input("btn-close-kontostand", "n_clicks"),
    State("kontostand-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_kontostand_modal(n1, n2, is_open):
    return not is_open

# Kontostand Display und Einzahlung/Auszahlung (kombiniert)
@callback(
    Output("kontostand-display", "children"),
    Output("balance-message", "children"),
    Input("kontostand-modal", "is_open"),
    Input("btn-deposit", "n_clicks"),
    Input("btn-withdraw", "n_clicks"),
    State("balance-amount", "value"),
    prevent_initial_call=True
)
def handle_kontostand(is_open, btn_deposit, btn_withdraw, amount):
    triggered = ctx.triggered_id
    balance = load_balance()
    
    # Modal wurde geöffnet - nur Anzeige aktualisieren
    if triggered == "kontostand-modal":
        if is_open:
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), ""
        return "", ""
    
    # Einzahlung
    if triggered == "btn-deposit":
        if not amount or amount <= 0:
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert("Ungültiger Betrag", color="danger")
        balance += float(amount)
        save_balance(balance)
        return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert(f"Einzahlung von {amount:,.2f} USD erfolgreich!", color="success")
    
    # Auszahlung
    if triggered == "btn-withdraw":
        if not amount or amount <= 0:
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert("Ungültiger Betrag", color="danger")
        if balance < float(amount):
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert("Nicht genügend Guthaben", color="danger")
        balance -= float(amount)
        save_balance(balance)
        return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert(f"Auszahlung von {amount:,.2f} USD erfolgreich!", color="success")
    
    raise dash.exceptions.PreventUpdate

# Buy Search
@callback(
    Output("buy-search-results", "children"),
    Output("search-results-store", "data"),
    Input("buy-search", "value"),
    prevent_initial_call=True
)
def search_for_buy(query):
    if not query or len(query) < 2:
        return [], []
    results = search_stocks(query)
    buttons = [
        dbc.Button(f"{r['symbol']} - {r['name']}", id={"type": "search-result", "index": i}, 
                   color="light", className="w-100 mb-1 text-start", size="sm")
        for i, r in enumerate(results[:5])
    ]
    return buttons, results[:5]


# (Balance displayed/updated by calculate_total)

# Select Stock for Buy
@callback(
    Output("buy-stock-info", "children"),
    Output("buy-chart", "figure"),
    Output("selected-ticker", "data"),
    Input({"type": "search-result", "index": dash.ALL}, "n_clicks"),
    State("search-results-store", "data"),
    prevent_initial_call=True
)
def select_stock_for_buy(clicks, results):
    if not any(clicks) or not results:
        return "", go.Figure(), None
    
    idx = next((i for i, c in enumerate(clicks) if c), 0)
    if idx >= len(results):
        return "", go.Figure(), None
    
    stock = results[idx]
    symbol = stock["symbol"]
    
    price, prev = fetch_price(symbol)
    if price:
        color = "#22c55e" if prev and price >= prev else "#ef4444" if prev else "#000"
        change = ""
        if prev:
            diff = price - prev
            pct = (diff / prev) * 100
            sign = "+" if diff >= 0 else ""
            change = f" ({sign}{diff:.2f} / {sign}{pct:.2f}%)"
        info = html.Div([
            html.H5(f"{stock['name']} ({symbol})"),
            html.H4(f"{price:.2f} USD{change}", style={"color": color})
        ])
    else:
        info = html.Div([html.H5(f"{stock['name']} ({symbol})"), html.P("Preis nicht verfügbar")])
        price = 0
    
    fig = create_stock_chart(symbol, "1d", "5m")
    return info, fig, {"symbol": symbol, "name": stock["name"], "price": price}

# Calculate Total and enforce balance
@callback(
    Output("buy-total", "children"),
    Output("buy-balance", "children"),
    Output("btn-confirm-buy", "disabled"),
    Input("buy-qty", "value"),
    Input("selected-ticker", "data"),
    Input("portfolio-store", "data"),
    prevent_initial_call=True
)
def calculate_total(qty, ticker, portfolio):

    # load current balance
    balance = load_balance()

    # If no ticker or qty, still show balance but no total
    if not ticker or not qty or not ticker.get("price"):
        balance_html = html.Span(f"Kontostand: {balance:,.2f} USD")
        return "", balance_html, True

    total = qty * ticker["price"]
    disabled = total > balance
    total_html = html.H5(f"Gesamt: {total:,.2f} USD")
    balance_html = html.Span(f"Kontostand: {balance:,.2f} USD")
    return total_html, balance_html, disabled

# Confirm Buy
@callback(
    Output("portfolio-store", "data", allow_duplicate=True),
    Input("btn-confirm-buy", "n_clicks"),
    State("selected-ticker", "data"),
    State("buy-qty", "value"),
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def confirm_buy(n, ticker, qty, portfolio):
    if not n or not ticker or not qty or not ticker.get("price"):
        return portfolio or []

    # Safety: prevent buying more than current balance
    balance = load_balance()
    total_cost = int(qty) * float(ticker.get("price", 0))

    if total_cost > balance:
        # Not enough funds: do not modify portfolio
        return portfolio or []
    
    portfolio = portfolio or []
    
    # Prüfen ob schon vorhanden
    found = False
    for item in portfolio:
        if item["symbol"] == ticker["symbol"]:
            # Gewichteter Durchschnitt berechnen
            old_qty = item["qty"]
            old_price = item.get("buy_price") or item.get("avg_price", 0)
            new_qty = old_qty + int(qty)
            new_price = ((old_price * old_qty) + (ticker["price"] * int(qty))) / new_qty
            item["qty"] = new_qty
            item["buy_price"] = new_price
            item["avg_price"] = new_price
            found = True
            break
    
    if not found:
        portfolio.append({
            "symbol": ticker["symbol"],
            "qty": int(qty),
            "buy_price": ticker["price"],
            "avg_price": ticker["price"]
        })
    
    save_portfolio(portfolio)
    
    save_transaction({
        "timestamp": datetime.now().isoformat(),
        "type": "buy",
        "symbol": ticker["symbol"],
        "qty": int(qty),
        "price": ticker["price"]
    })
    
    # Kontostand anpassen: Betrag abziehen
    try:
        current_balance = load_balance()
        cost = int(qty) * float(ticker.get("price", 0))
        current_balance -= cost
        save_balance(current_balance)
    except Exception:
        # Falls Balance-Funktionen fehlen oder fehlschlagen, nichts weiter tun
        pass

    return portfolio

# Confirm Sell
@callback(
    Output("portfolio-store", "data", allow_duplicate=True),
    Input("btn-confirm-sell", "n_clicks"),
    State("selected-ticker", "data"),
    State("buy-qty", "value"),
    State("portfolio-store", "data"),
    prevent_initial_call=True
)
def confirm_sell(n, ticker, qty, portfolio):
    if not n or not ticker or not qty:
        return portfolio or []
    
    symbol = ticker["symbol"]
    qty = int(qty)
    portfolio = portfolio or []
    
    # Finde Position im Portfolio
    for item in portfolio:
        if item["symbol"] == symbol:
            if item["qty"] >= qty:
                item["qty"] -= qty
                if item["qty"] == 0:
                    portfolio.remove(item)
                break
    
    save_portfolio(portfolio)
    
    save_transaction({
        "timestamp": datetime.now().isoformat(),
        "type": "sell",
        "symbol": symbol,
        "qty": qty,
        "price": ticker.get("price", 0)
    })
    
    # Kontostand anpassen: Erlös hinzufügen
    try:
        current_balance = load_balance()
        proceeds = qty * float(ticker.get("price", 0))
        current_balance += proceeds
        save_balance(current_balance)
    except Exception:
        pass

    return portfolio

# Transactions Modal
@callback(
    Output("transactions-modal", "is_open"),
    Output("transactions-table", "children"),
    Output("transactions-summary", "children"),
    Output("tx-year", "options"),
    Input("btn-transactions", "n_clicks"),
    Input("btn-close-tx", "n_clicks"),
    Input("tx-year", "value"),
    Input("tx-month", "value"),
    Input("tx-type", "value"),
    State("transactions-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_transactions(n1, n2, year, month, tx_type, is_open):
    triggered = ctx.triggered_id
    
    if triggered in ["btn-transactions", "btn-close-tx"]:
        is_open = not is_open
    
    txs = load_transactions()
    
    # Jahr-Optionen
    years = sorted(set(datetime.fromisoformat(t["timestamp"]).year for t in txs), reverse=True) if txs else []
    year_options = [{"label": "Alle Jahre", "value": "all"}] + [{"label": str(y), "value": str(y)} for y in years]
    
    # Filtern
    filtered = txs
    if year and year != "all":
        filtered = [t for t in filtered if datetime.fromisoformat(t["timestamp"]).year == int(year)]
    if month and month != "all":
        filtered = [t for t in filtered if datetime.fromisoformat(t["timestamp"]).month == int(month)]
    if tx_type and tx_type != "all":
        filtered = [t for t in filtered if t["type"] == tx_type]
    
    if not filtered:
        return is_open, html.P("Keine Transaktionen vorhanden", className="text-muted"), "", year_options
    
    rows = []
    total_buy = 0
    total_sell = 0
    
    for t in sorted(filtered, key=lambda x: x["timestamp"], reverse=True):
        dt = datetime.fromisoformat(t["timestamp"])
        total = t["qty"] * t["price"]
        if t["type"] == "buy":
            total_buy += total
        else:
            total_sell += total
        
        rows.append({
            "Datum": dt.strftime("%d.%m.%Y"),
            "Zeit": dt.strftime("%H:%M"),
            "Typ": "Kauf" if t["type"] == "buy" else "Verkauf",
            "Symbol": t["symbol"],
            "Menge": t["qty"],
            "Kurs": f"{t['price']:.2f}",
            "Gesamt": f"{total:.2f}"
        })
    
    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["Datum", "Zeit", "Typ", "Symbol", "Menge", "Kurs", "Gesamt"]],
        style_cell={"textAlign": "center", "padding": "8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": "{Typ} = 'Kauf'"}, "backgroundColor": "#dcfce7"},
            {"if": {"filter_query": "{Typ} = 'Verkauf'"}, "backgroundColor": "#fee2e2"},
        ]
    )
    
    saldo = total_sell - total_buy
    summary = dbc.Alert([
        f"Transaktionen: {len(filtered)} | ",
        html.Span(f"Käufe: {total_buy:,.2f} USD", style={"color": "#22c55e"}),
        " | ",
        html.Span(f"Verkäufe: {total_sell:,.2f} USD", style={"color": "#ef4444"}),
        f" | Saldo: {saldo:+,.2f} USD"
    ], color="light")
    
    return is_open, table, summary, year_options

# Ticker Detail Modal (Klick auf Market Overview)
@callback(
    Output("ticker-modal", "is_open"),
    Output("ticker-modal-header", "children"),
    Output("ticker-modal-stats", "children"),
    Output("ticker-modal-chart", "figure"),
    Output("current-ticker-symbol", "data"),
    [Input(f"ticker-{s['name']}", "n_clicks") for s in MARKET_OVERVIEW_SYMBOLS] +
    [Input("btn-close-ticker", "n_clicks"),
     Input("ticker-btn-1d", "n_clicks"),
     Input("ticker-btn-1w", "n_clicks"),
     Input("ticker-btn-1m", "n_clicks"),
     Input("ticker-btn-3m", "n_clicks")],
    State("ticker-modal", "is_open"),
    State("current-ticker-symbol", "data"),
    prevent_initial_call=True
)
def toggle_ticker_modal(*args):
    # Parse arguments
    num_tickers = len(MARKET_OVERVIEW_SYMBOLS)
    ticker_clicks = args[:num_tickers]
    close_click = args[num_tickers]
    period_clicks = args[num_tickers+1:num_tickers+5]
    is_open = args[-2]
    current_symbol = args[-1]
    
    triggered = ctx.triggered_id
    
    if triggered == "btn-close-ticker":
        return False, "", "", go.Figure(), None
    
    # Zeitraum-Buttons
    period_map = {
        "ticker-btn-1d": ("1d", "5m"),
        "ticker-btn-1w": ("5d", "15m"),
        "ticker-btn-1m": ("1mo", "1d"),
        "ticker-btn-3m": ("3mo", "1d"),
    }
    
    if triggered in period_map and current_symbol:
        period, interval = period_map[triggered]
        fig = create_stock_chart(current_symbol["symbol"], period, interval)
        return True, current_symbol["header"], current_symbol["stats"], fig, current_symbol
    
    # Finde geklickten Ticker
    for i, s in enumerate(MARKET_OVERVIEW_SYMBOLS):
        if triggered == f"ticker-{s['name']}":
            symbol = s["symbol"]
            name = s["name"]
            
            price, prev = fetch_price(symbol)
            if s.get("invert") and price:
                price = 1 / price
                if prev:
                    prev = 1 / prev
            
            # Stats
            try:
                t = yf.Ticker(symbol)
                fast = t.fast_info if hasattr(t, "fast_info") else {}
                high = getattr(fast, "day_high", None)
                low = getattr(fast, "day_low", None)
                vol = getattr(fast, "last_volume", None)
            except:
                high, low, vol = None, None, None
            
            price_text = f"{price:.4f}" if price else "n/a"
            high_text = f"{high:.2f}" if high else "n/a"
            low_text = f"{low:.2f}" if low else "n/a"
            
            stats = html.Div([
                dbc.Row([
                    dbc.Col([html.B("Kurs: "), price_text], width=3),
                    dbc.Col([html.B("High: "), high_text], width=3),
                    dbc.Col([html.B("Low: "), low_text], width=3),
                    dbc.Col([html.B("Vol: "), format_volume(vol)], width=3),
                ])
            ])
            
            fig = create_stock_chart(symbol, "1d", "5m")
            header = f"{name} ({symbol})"
            
            # Speichere Symbol-Info für Zeitraum-Wechsel
            symbol_data = {"symbol": symbol, "name": name, "header": header, "stats": stats}
            
            return True, header, stats, fig, symbol_data
    
    return is_open, "", "", go.Figure(), current_symbol

# Sentiment Stock Search
@callback(
    Output("sentiment-stock-dropdown", "options"),
    Output("sentiment-stock-dropdown", "value"),
    Input("sentiment-search-input", "value"),
    prevent_initial_call=True
)
def sentiment_search_stocks(search_query):
    if not search_query or len(search_query) < 2:
        return [], None
    
    results = search_stocks(search_query)
    if not results:
        return [{"label": "Keine Ergebnisse gefunden", "value": "", "disabled": True}], None
    
    options = [
        {"label": f"{r['name']} ({r['symbol']}) - {r['exchange']}", "value": r['symbol']}
        for r in results
    ]
    
    # Wenn nur ein Ergebnis, direkt auswählen
    default_value = results[0]['symbol'] if len(results) == 1 else None
    
    return options, default_value

# Sentiment Analysis
@callback(
    Output("ai-sentiment-output", "children"),
    Input("btn-sentiment-analyze", "n_clicks"),
    State("sentiment-stock-dropdown", "value"),
    State("sentiment-period-select", "value"),
    State("sentiment-news-count", "value"),
    prevent_initial_call=True
)
def sentiment_analyze(n_clicks, symbol, period, news_count):
    if not symbol:
        return dbc.Alert("Bitte wählen Sie eine Aktie aus der Dropdown-Liste aus.", color="warning")
    
    symbol = symbol.strip().upper()
    news_limit = int(news_count) if news_count else 30
    
    if not VADER_AVAILABLE:
        return dbc.Alert("Die vaderSentiment-Bibliothek ist nicht installiert. Bitte installieren Sie sie mit: pip install vaderSentiment", color="danger")
    
    try:
        # Zeitraum in Tage für News-Filter
        period_days = {
            "1d": 1, "5d": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825
        }
        days_back = period_days.get(period, 30)
        cutoff_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).date()
        
        # 1. RSS-Feeds abrufen (erweiterte Quellen für mehr historische Daten)
        rss_feeds = [
            f"https://news.google.com/rss/search?q={symbol}+stock&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=DE&ceid=DE:de",
            f"https://news.google.com/rss/search?q={symbol}+shares&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+investor&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+earnings&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+quarterly+report&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+market&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+CEO&hl=en&gl=US&ceid=US:en",
            "https://www.wallstreet-online.de/rss/nachrichten-aktien-indizes.xml",
            "https://www.ft.com/rss/home/international",
            "https://feeds.bloomberg.com/markets/news.rss",
        ]
        
        analyzer = SentimentIntensityAnalyzer()
        news_data = []
        news_items = []
        sources_found = []
        seen_titles = set()  # Duplikate vermeiden
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                if feed.entries:
                    # Quelle identifizieren
                    if "google.com" in feed_url:
                        source_name = "Google News"
                    elif "wallstreet-online" in feed_url:
                        source_name = "Wallstreet Online"
                    elif "ft.com" in feed_url:
                        source_name = "Financial Times"
                    elif "bloomberg" in feed_url:
                        source_name = "Bloomberg"
                    else:
                        source_name = "Unbekannt"
                    
                    entries_count = 0
                    for entry in feed.entries:
                        # Bei allgemeinen Feeds: nur relevante News (mit Symbol im Titel)
                        title = entry.get("title", "")
                        
                        # Duplikate überspringen
                        title_hash = title.lower().strip()[:50]
                        if title_hash in seen_titles:
                            continue
                        seen_titles.add(title_hash)
                        
                        if "wallstreet-online" in feed_url or "ft.com" in feed_url or "bloomberg" in feed_url:
                            # Prüfe ob Symbol oder Firmenname im Titel
                            if symbol.lower() not in title.lower():
                                continue
                        
                        try:
                            # Publikationsdatum parsen und filtern
                            pub_date = entry.get("published") or entry.get("pubDate") or entry.get("updated")
                            if pub_date:
                                date = pd.to_datetime(pub_date)
                                # Nur News im gewählten Zeitraum
                                if date.date() < cutoff_date:
                                    continue
                            else:
                                date = pd.Timestamp.now()
                            
                            text = title
                            score = analyzer.polarity_scores(text)["compound"]
                            news_data.append({"date": date.date(), "score": score, "title": text, "source": source_name})
                            news_items.append({"title": text, "score": score, "date": date.strftime("%d.%m.%Y"), "source": source_name})
                            entries_count += 1
                            
                            if len(news_data) >= news_limit:
                                break
                        except:
                            continue
                    
                    if entries_count > 0:
                        sources_found.append(f"{source_name} ({entries_count})")
            except:
                continue
            
            if len(news_data) >= news_limit:
                break
        
        if not news_data:
            return dbc.Alert(f"Keine News für '{symbol}' in den RSS-Feeds gefunden.", color="warning")
        
        # 2. DataFrame erstellen und pro Tag mitteln
        news_df = pd.DataFrame(news_data)
        sentiment_daily = news_df.groupby("date")["score"].mean()
        
        # 3. Kursdaten abrufen
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period or "1mo")
        
        if hist.empty:
            return dbc.Alert(f"Keine Kursdaten für '{symbol}' verfügbar.", color="danger")
        
        # 4. Plotly Chart erstellen (Dual-Axis)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Kurslinie
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        is_positive = end_price >= start_price
        color_line = "#22c55e" if is_positive else "#ef4444"
        
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
        
        # Durchschnittlichen Sentiment berechnen
        avg_sentiment = news_df["score"].mean()
        sentiment_label = "positiv" if avg_sentiment > 0.05 else "negativ" if avg_sentiment < -0.05 else "neutral"
        sentiment_color = "success" if avg_sentiment > 0.05 else "danger" if avg_sentiment < -0.05 else "secondary"
        
        # News-Liste erstellen (Top 5)
        news_list = []
        for item in sorted(news_items, key=lambda x: abs(x["score"]), reverse=True)[:5]:
            score_badge = dbc.Badge(
                f"{item['score']:.2f}",
                color="success" if item["score"] > 0 else "danger" if item["score"] < 0 else "secondary",
                className="me-2"
            )
            source_badge = dbc.Badge(
                item.get("source", ""),
                color="light",
                text_color="dark",
                className="me-2"
            )
            news_list.append(
                html.Li([
                    score_badge,
                    source_badge,
                    html.Small(f"[{item['date']}] ", className="text-muted"),
                    item["title"]
                ], className="mb-2", style={"fontSize": "0.9rem"})
            )
        
        # Quellen-Info erstellen
        sources_info = ", ".join(sources_found) if sources_found else "Keine Quellen"
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Durchschnittlicher Sentiment", className="card-title"),
                            html.H2(f"{avg_sentiment:.2f}", className=f"text-{sentiment_color}"),
                            dbc.Badge(sentiment_label.upper(), color=sentiment_color, className="mt-2")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Analysierte News", className="card-title"),
                            html.H2(f"{len(news_items)}", className="text-primary"),
                            html.Small("Artikel analysiert", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Kursänderung", className="card-title"),
                            html.H2(f"{sign}{pct_change:.2f}%", className=f"text-{'success' if is_positive else 'danger'}"),
                            html.Small(f"{start_price:.2f} → {end_price:.2f} USD", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Sentiment-Tage", className="card-title"),
                            html.H2(f"{len(sentiment_daily)}", className="text-info"),
                            html.Small("Tage mit News", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
            ], className="mb-3"),
            dbc.Alert([
                html.Strong("📡 Quellen: "),
                sources_info
            ], color="light", className="mb-3"),
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H6("📰 Top News nach Sentiment-Stärke:"),
            html.Ul(news_list, style={"listStyleType": "none", "paddingLeft": "0"})
        ])
        
    except Exception as e:
        return dbc.Alert(f"Fehler bei der Analyse: {str(e)}", color="danger")

# Forecast Analysis
@callback(
    Output("ai-forecast-output", "children"),
    Input("btn-forecast-analyze", "n_clicks"),
    State("ai-forecast-input", "value"),
    prevent_initial_call=True
)
def forecast_analyze(n_clicks, input_text):
    if not input_text:
        return html.P("Bitte geben Sie eine Aktie oder einen Markt ein.", className="text-muted")
    
    # Dummy Forecast Response
    response = f"🔮 Prognose für: '{input_text}'\n\nBasierend auf historischen Daten und aktuellen Trends wird eine moderate Steigerung in den nächsten Wochen erwartet. Risiken bestehen durch Marktvolatilität."
    
    return dbc.Alert(response, color="warning")

# Correlation Stock Search
@callback(
    Output("corr-stock-dropdown", "options"),
    Output("corr-stock-dropdown", "value"),
    Input("corr-search-input", "value"),
    prevent_initial_call=True
)
def corr_search_stocks(search_query):
    if not search_query or len(search_query) < 2:
        return [], None
    
    results = search_stocks(search_query)
    if not results:
        return [{"label": "Keine Ergebnisse gefunden", "value": "", "disabled": True}], None
    
    options = [
        {"label": f"{r['name']} ({r['symbol']}) - {r['exchange']}", "value": r['symbol']}
        for r in results
    ]
    
    default_value = results[0]['symbol'] if len(results) == 1 else None
    
    return options, default_value

# Correlation Analysis
@callback(
    Output("corr-output", "children"),
    Input("btn-corr-analyze", "n_clicks"),
    State("corr-stock-dropdown", "value"),
    State("corr-period-select", "value"),
    prevent_initial_call=True
)
def correlation_analyze(n_clicks, symbol, period):
    if not symbol:
        return dbc.Alert("Bitte wählen Sie eine Aktie aus der Dropdown-Liste aus.", color="warning")
    
    symbol = symbol.strip().upper()
    
    if not VADER_AVAILABLE:
        return dbc.Alert("Die vaderSentiment-Bibliothek ist nicht installiert.", color="danger")
    
    try:
        # Zeitraum in Tage umrechnen für News-Filter
        period_days = {
            "5d": 7,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "5y": 1825
        }
        days_back = period_days.get(period, 90)
        cutoff_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).date()
        
        # RSS-Feeds mit erweiterten historischen Suchoptionen
        rss_feeds = [
            f"https://news.google.com/rss/search?q={symbol}+stock&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=DE&ceid=DE:de",
            f"https://news.google.com/rss/search?q={symbol}+shares&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+investor&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+earnings&hl=en&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={symbol}+quarterly&hl=en&gl=US&ceid=US:en",
        ]
        
        analyzer = SentimentIntensityAnalyzer()
        news_data = []
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    try:
                        pub_date = entry.get("published") or entry.get("pubDate") or entry.get("updated")
                        if pub_date:
                            date = pd.to_datetime(pub_date)
                            if date.date() < cutoff_date:
                                continue
                        else:
                            continue
                        
                        title = entry.get("title", "")
                        score = analyzer.polarity_scores(title)["compound"]
                        news_data.append({"date": date.date(), "score": score})
                    except:
                        continue
            except:
                continue
        
        if len(news_data) < 5:
            return dbc.Alert(f"Zu wenige News für '{symbol}' gefunden ({len(news_data)} Artikel). Versuchen Sie einen kürzeren Zeitraum.", color="warning")
        
        # DataFrame und täglicher Durchschnitt
        news_df = pd.DataFrame(news_data)
        sentiment_daily = news_df.groupby("date")["score"].mean().reset_index()
        sentiment_daily.columns = ["date", "sentiment"]
        sentiment_daily["date"] = pd.to_datetime(sentiment_daily["date"])
        
        # Kursdaten abrufen
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period or "3mo")
        
        if hist.empty:
            return dbc.Alert(f"Keine Kursdaten für '{symbol}' verfügbar.", color="danger")
        
        # Kursdaten vorbereiten
        price_df = hist[["Close"]].reset_index()
        price_df.columns = ["date", "price"]
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)
        
        # Merge: Sentiment mit Kursdaten
        merged_df = pd.merge(price_df, sentiment_daily, on="date", how="left")
        merged_df["sentiment"] = merged_df["sentiment"].interpolate(method="linear").fillna(0)
        
        # Korrelation berechnen
        correlation = merged_df["price"].corr(merged_df["sentiment"])
        
        # Normalisierte Werte für bessere Visualisierung
        merged_df["price_norm"] = (merged_df["price"] - merged_df["price"].min()) / (merged_df["price"].max() - merged_df["price"].min())
        merged_df["sentiment_norm"] = (merged_df["sentiment"] - merged_df["sentiment"].min()) / (merged_df["sentiment"].max() - merged_df["sentiment"].min() + 0.001)
        
        # Rolling Average für Sentiment (7 Tage)
        merged_df["sentiment_ma"] = merged_df["sentiment"].rolling(window=7, min_periods=1).mean()
        
        # Plotly Chart erstellen
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f"{symbol} Kurs mit Sentiment-Overlay", "Sentiment-Score (7-Tage Durchschnitt)")
        )
        
        # Kurs-Linie
        start_price = merged_df["price"].iloc[0]
        end_price = merged_df["price"].iloc[-1]
        is_positive = end_price >= start_price
        color_price = "#22c55e" if is_positive else "#ef4444"
        
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
        
        # Sentiment-Punkte auf Kurschart (farbcodiert)
        sentiment_colors = ["#22c55e" if s > 0.05 else "#ef4444" if s < -0.05 else "#6b7280" for s in merged_df["sentiment"]]
        
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
        pct_change = ((end_price - start_price) / start_price) * 100
        corr_color = "success" if correlation > 0.3 else "danger" if correlation < -0.3 else "secondary"
        corr_label = "Stark positiv" if correlation > 0.5 else "Positiv" if correlation > 0.3 else "Stark negativ" if correlation < -0.5 else "Negativ" if correlation < -0.3 else "Schwach/Neutral"
        
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
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Korrelationskoeffizient", className="card-title"),
                            html.H2(f"{correlation:.3f}", className=f"text-{corr_color}"),
                            dbc.Badge(corr_label, color=corr_color, className="mt-2")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("News analysiert", className="card-title"),
                            html.H2(f"{len(news_data)}", className="text-primary"),
                            html.Small(f"über {days_back} Tage", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Kursänderung", className="card-title"),
                            html.H2(f"{'+' if pct_change >= 0 else ''}{pct_change:.2f}%", className=f"text-{'success' if is_positive else 'danger'}"),
                            html.Small(f"{start_price:.2f} → {end_price:.2f}", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Ø Sentiment", className="card-title"),
                            html.H2(f"{merged_df['sentiment'].mean():.3f}", className=f"text-{'success' if merged_df['sentiment'].mean() > 0 else 'danger'}"),
                            html.Small("Durchschnitt", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
            ], className="mb-3"),
            dbc.Alert([
                html.Strong("📊 Interpretation: "),
                f"Ein Korrelationskoeffizient von {correlation:.3f} bedeutet ",
                html.Strong("starke positive Beziehung" if correlation > 0.5 else "moderate positive Beziehung" if correlation > 0.3 else "starke negative Beziehung" if correlation < -0.5 else "moderate negative Beziehung" if correlation < -0.3 else "schwache/keine lineare Beziehung"),
                " zwischen Sentiment und Kursbewegung. ",
                "Grüne Hintergründe = positive Stimmung, Rote = negative Stimmung."
            ], color="info", className="mb-3"),
            dcc.Graph(figure=fig),
        ])
        
    except Exception as e:
        return dbc.Alert(f"Fehler bei der Korrelationsanalyse: {str(e)}", color="danger")

# ============== Server starten ==============
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Stock Dashboard startet...")
    print("📊 Öffne im Browser: http://localhost:8050")
    print("=" * 50)
    app.run(debug=True, port=8050)
