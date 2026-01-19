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
import pandas as pd

from sentiment_analysis import (
    VADER_AVAILABLE, analyze_sentiment, analyze_correlation, get_sentiment_label,
    get_correlation_label, ARIMA_AVAILABLE, analyze_forecast, get_forecast_label,
    analyze_monte_carlo, get_monte_carlo_label,
)

# KONFIGURATION: Dateipfade und Marktübersicht
DATA_DIR = Path(__file__).parent / "gui"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRANSACTIONS_FILE = DATA_DIR / "transactions.json"
BALANCE_FILE = DATA_DIR / "balance.json"

MARKET_OVERVIEW_SYMBOLS = [
    {"name": "DAX", "symbol": "^GDAXI", "decimals": 0},
    {"name": "MDAX", "symbol": "^MDAXI", "decimals": 0},
    {"name": "SDAX", "symbol": "^SDAXI", "decimals": 0},
    {"name": "Dow", "symbol": "^DJI", "decimals": 0},
    {"name": "Nasdaq", "symbol": "^IXIC", "decimals": 0},
    {"name": "Gold", "symbol": "GC=F", "decimals": 2},
    {"name": "Brent", "symbol": "BZ=F", "decimals": 2},
    {"name": "BTC", "symbol": "BTC-USD", "decimals": 0},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "decimals": 4},
]

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
    
    y_min = hist["Close"].min()
    y_max = hist["Close"].max()
    y_range = y_max - y_min
    padding = 0.005 * y_max if y_range < 0.01 * y_max else y_range * 0.1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({34 if is_positive else 239}, {197 if is_positive else 68}, {94 if is_positive else 68}, 0.1)",
        name=symbol, hovertemplate="%{y:.2f}<extra></extra>"
    ))
    
    pct_change = ((end_price - start_price) / start_price) * 100
    sign = "+" if pct_change >= 0 else ""
    
    fig.update_layout(
        title=dict(text=f"{symbol} ({sign}{pct_change:.2f}%)", font=dict(size=16, color=color)),
        yaxis=dict(range=[y_min - padding, y_max + padding], tickformat=",.2f", gridcolor="#e5e7eb"),
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode="x unified", showlegend=False
    )
    return fig

def create_portfolio_pie_chart(portfolio):
    if not portfolio:
        fig = go.Figure()
        fig.add_annotation(text="Portfolio ist leer", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    try:
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
                colors.append(f"hsl({hash(symbol) % 360}, 70%, 50%)")
        
        if not values:
            fig = go.Figure()
            fig.add_annotation(text="Keine aktuellen Preise verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, marker_colors=colors,
            textinfo='label+percent', insidetextorientation='radial',
            textfont=dict(color="white")
        )])
        
        fig.update_layout(
            title=dict(text="Portfolio-Zusammensetzung", font=dict(color="white")),
            showlegend=False, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Fehler beim Laden", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

def create_portfolio_value_chart(portfolio):
    
    def empty_fig(text):
        fig = go.Figure()
        fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    if not portfolio:
        return empty_fig("Portfolio ist leer")
    
    try:
        data_list = []
        total_invested = 0
        total_current = 0
        
        for item in portfolio:
            symbol = item.get("symbol", "")
            qty = item.get("qty", 0)
            buy_price = item.get("buy_price") or item.get("avg_price", 0)
            invested = qty * buy_price
            total_invested += invested
            
            try:
                current_price, _ = fetch_price(symbol)
            except:
                current_price = None
            
            if current_price:
                current_value = qty * current_price
                total_current += current_value
                pnl = current_value - invested
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                
                try:
                    name = fetch_name(symbol) or symbol
                except:
                    name = symbol
                
                data_list.append({
                    "symbol": symbol,
                    "name": name,
                    "invested": invested,
                    "current": current_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "buy_price": buy_price,
                    "current_price": current_price,
                    "qty": qty
                })
        
        # Wenn keine Daten gesammelt wurden
        if not data_list:
            return empty_fig("Keine aktuellen Kursdaten verfügbar")
        
        # Sortiere alphabetisch nach Symbol
        data_list.sort(key=lambda x: x["symbol"])
        
        # Kürze lange Namen auf 15 Zeichen
        symbols = [d["name"][:15] if d["name"] else d["symbol"] for d in data_list]
        invested_values = [d["invested"] for d in data_list]
        current_values = [d["current"] for d in data_list]
        pnl_values = [d["pnl"] for d in data_list]
        
        fig = go.Figure()
        
        for i, d in enumerate(data_list):
            x_pos = [i - 0.3, i + 0.3, i + 0.3, i - 0.3]
            
            if d["current"] >= d["invested"]:
                y_fill = [d["invested"], d["invested"], d["current"], d["current"]]
                fig.add_trace(go.Scatter(
                    x=x_pos, y=y_fill,
                    fill="toself",  # Fülle die Form
                    fillcolor="rgba(34, 197, 94, 0.3)",  # Halbtransparentes Grün
                    line=dict(width=0),  # Keine Umrandung
                    showlegend=False,
                    hoverinfo="skip"  # Kein Tooltip für Füllung
                ))
            else:
                y_fill = [d["current"], d["current"], d["invested"], d["invested"]]
                fig.add_trace(go.Scatter(
                    x=x_pos, y=y_fill,
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.3)",  # Halbtransparentes Rot
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(symbols))),  # X-Positionen: 0, 1, 2, ...
            y=invested_values,
            mode="lines+markers",          # Linie mit Punkten
            name="Kaufwert",
            line=dict(color="#fbbf24", width=3, dash="dash"),  # Gelb, gestrichelt
            marker=dict(size=10, symbol="diamond"),  # Diamant-Marker
            hovertemplate="<b>Kaufwert</b><br>%{y:,.2f} USD<extra></extra>"
        ))
        
        # Farbe der Punkte: Grün bei Gewinn, Rot bei Verlust
        colors = ["#22c55e" if pnl >= 0 else "#ef4444" for pnl in pnl_values]
        
        fig.add_trace(go.Scatter(
            x=list(range(len(symbols))),
            y=current_values,
            mode="lines+markers",
            name="Aktueller Wert",
            line=dict(color="#3b82f6", width=3),  # Blau
            marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
            hovertemplate="<b>Aktueller Wert</b><br>%{y:,.2f} USD<extra></extra>"
        ))
        
        annotations = []
        for i, d in enumerate(data_list):
            color = "#22c55e" if d["pnl"] >= 0 else "#ef4444"
            sign = "+" if d["pnl"] >= 0 else ""
            
            # Platziere Text über dem höchsten Punkt
            annotations.append(dict(
                x=i,
                y=max(d["current"], d["invested"]) * 1.08,  # 8% über dem höchsten Wert
                text=f"<b>{sign}{d['pnl']:.0f}$</b><br><span style='font-size:10px'>({sign}{d['pnl_pct']:.1f}%)</span>",
                showarrow=False,
                font=dict(color=color, size=11),
                align="center"
            ))
        
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        total_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        total_sign = "+" if total_pnl >= 0 else ""
        
        fig.update_layout(
            # Titel mit Gesamt-P/L
            title=dict(
                text=f"Portfolio-Wertentwicklung | Gesamt: <span style='color:{total_color}'>{total_sign}{total_pnl:,.2f}$ ({total_sign}{total_pnl_pct:.1f}%)</span>",
                font=dict(size=14, color="white")
            ),
            # X-Achse: Aktien-Namen als Beschriftung
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(symbols))),
                ticktext=symbols,
                tickfont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            ),
            # Y-Achse: Wert in USD
            yaxis=dict(
                title="Wert (USD)",
                tickformat=",.0f",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            ),
            annotations=annotations,  # P/L-Texte hinzufügen
            # Legende horizontal oben
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color="white")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,30,30,0.5)",
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode="x unified"
        )
        
        return fig
        
    except Exception as e:
        return empty_fig(f"Fehler: {str(e)[:50]}")

# APP INITIALISIERUNG
# Hier wird die Dash-Anwendung erstellt und konfiguriert.
# Dash ist das Framework, das unsere Web-App antreibt.

# Erstelle die Dash-App
app = dash.Dash(
    __name__,  # Name des Moduls (wird für Pfade verwendet)
    external_stylesheets=[
        dbc.themes.DARKLY,  # Bootstrap Dark Theme für ein modernes dunkles Design
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ], 
    suppress_callback_exceptions=True  # Unterdrückt Fehler bei dynamischen Callbacks
)

# Titel der Webseite (erscheint im Browser-Tab)
app.title = "Stock Dashboard"

# CUSTOM CSS (Benutzerdefinierte Styles)
# Hier fügen wir eigenes CSS hinzu für:
# - News-Kacheln mit Hover-Effekten
# - Light Mode / Dark Mode Unterstützung
# - Benutzerdefinierte Tabellenformatierung
# - Theme-Toggle-Button-Animation
#
# Die Syntax ist: selector { eigenschaft: wert; }

    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* =================================================
               NEWS CARD STYLES - Styling für Nachrichtenkacheln
               ================================================= */
            
            /* Hover-Effekt: Karte hebt sich an und bekommt Schatten */
            .news-card:hover {
                transform: translateY(-5px);  /* 5px nach oben verschieben */
                box-shadow: 0 8px 25px rgba(0, 123, 255, 0.3) !important;  /* Blauer Schatten */
            }
            
            /* Mindesthhe für den News-Container */
            .news-grid-container {
                min-height: 60vh;  /* 60% der Viewport-Höhe */
            }
            
            /* Hintergrund-Farbverlauf für News-Karten */
            .news-card .card-body {
                background: linear-gradient(180deg, #2d3436 0%, #1e272e 100%);
            }
            
            /* Bilder in News-Karten etwas abdunkeln */
            .news-card img {
                filter: brightness(0.9);  /* 90% Helligkeit */
                transition: filter 0.3s;   /* Sanfte Animation */
            }
            
            /* Bei Hover: Bild aufhellen */
            .news-card:hover img {
                filter: brightness(1.1);  /* 110% Helligkeit */
            }
            
            /* =================================================
               LIGHT MODE OVERRIDES - Helles Design
               ================================================= 
               Diese Regeln werden aktiv, wenn die body-Klasse
               "light-mode" gesetzt ist (durch JavaScript).
            */
            body.light-mode {
                background-color: #f8f9fa !important;  /* Hellgrauer Hintergrund */
                color: #212529 !important;              /* Dunkler Text */
            }
            body.light-mode .bg-dark {
                background-color: #ffffff !important;  /* Dunkle Elemente werden weiß */
            }
            body.light-mode .text-white {
                color: #212529 !important;             /* Weißer Text wird dunkel */
            }
            body.light-mode .card {
                background-color: #ffffff !important;
                border-color: #dee2e6 !important;
            }
            body.light-mode .news-card .card-body {
                background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
            }
            body.light-mode .news-card .text-white {
                color: #212529 !important;
            }
            body.light-mode .table {
                color: #212529 !important;
            }
            body.light-mode .nav-tabs .nav-link {
                color: #495057 !important;
            }
            body.light-mode .nav-tabs .nav-link.active {
                background-color: #ffffff !important;
                color: #212529 !important;
            }
            body.light-mode .container-fluid,
            body.light-mode .container {
                background-color: #f8f9fa !important;
            }
            body.light-mode .form-control,
            body.light-mode .form-select {
                background-color: #ffffff !important;
                color: #212529 !important;
                border-color: #ced4da !important;
            }
            body.light-mode .input-group-text {
                background-color: #e9ecef !important;
                color: #212529 !important;
            }
            body.light-mode .modal-content {
                background-color: #ffffff !important;
            }
            body.light-mode .text-muted {
                color: #6c757d !important;
            }
            
            /* =================================================
               PORTFOLIO TABLE STYLES - Tabellen-Formatierung
               ================================================= */
            .portfolio-table th {
                font-weight: 600;  /* Fette Überschriften */
            }
            .portfolio-table td {
                vertical-align: middle;  /* Vertikale Zentrierung */
            }
            
            /* =================================================
               THEME TOGGLE BUTTON - Animation für Theme-Wechsel
               ================================================= */
            .theme-toggle-btn {
                transition: all 0.3s ease;  /* Sanfte Animation bei allen Änderungen */
            }
            .theme-toggle-btn:hover {
                transform: scale(1.1);  /* Bei Hover: 10% vergrößern */
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# LAYOUT - Die visuelle Struktur der Anwendung
#
# Wichtige Konzepte:
# - dbc.Row/dbc.Col: Zeilen und Spalten für das Grid-Layout
# - dbc.Button: Ein klickbarer Button
# - dcc.Input: Ein Eingabefeld
# - dcc.Graph: Ein Plotly-Diagramm
# - dbc.Modal: Ein Pop-up Fenster

def create_market_ticker():
    Diese Leiste zeigt die wichtigsten Marktindizes und deren aktuelle Kurse an.
    Jedes Element ist klickbar und öffnet ein Detail-Modal.
    
    Rückgabe: Eine dbc.Row mit den Ticker-Elementen
    """
    return dbc.Row([
        dbc.Col(
            html.Div(
                id=f"ticker-{s['name']}",  # Eindeutige ID für Callbacks
                className="text-center p-2",  # Bootstrap-Klassen: zentriert, Padding
                style={
                    "cursor": "pointer",          # Mauszeiger zeigt an, dass klickbar
                    "borderRadius": "5px",        # Abgerundete Ecken
                    "background": "rgba(128, 128, 128, 0.2)"  # Halbtransparentes Grau
                }
            ), 
            width="auto"  # Breite passt sich dem Inhalt an
        ) 
        for s in MARKET_OVERVIEW_SYMBOLS  # Durchlaufe alle Symbole
    ], 
    className="g-2 p-2 mb-3",  # g-2 = Gap/Abstand, p-2 = Padding, mb-3 = Margin Bottom
    justify="center",  # Elemente zentrieren
    style={"background": "rgba(128, 128, 128, 0.1)", "borderRadius": "8px"}  # Hintergrund
    )

# HAUPT-LAYOUT DER ANWENDUNG
# Hier werden alle UI-Elemente definiert und angeordnet.

app.layout = dbc.Container([
    dcc.Interval(id="market-interval", interval=15000, n_intervals=0),  # 15000ms = 15 Sekunden
    
    # dcc.Store speichert Daten im Browser des Benutzers
    dcc.Store(id="selected-ticker", data=None),           # Aktuell ausgewählte Aktie für Buy/Sell
    dcc.Store(id="portfolio-store", data=load_portfolio()),  # Portfolio-Daten (geladen aus Datei)
    dcc.Store(id="search-results-store", data=[]),        # Suchergebnisse
    dcc.Store(id="theme-store", data="dark"),             # Aktuelles Theme (dark/light)
    
    dbc.Row([
        # Linke Spalte: Titel
        dbc.Col([
            html.H4("📈 Stock Dashboard", className="mb-0"),
        ], width=8),
        
        # Rechte Spalte: Theme-Wechsel-Buttons
        dbc.Col([
            dbc.ButtonGroup([
                # Light Mode Button
                dbc.Button(
                    [html.I(className="fas fa-sun me-1"), "Light"],  # Sonnen-Icon + Text
                    id="btn-light-mode",
                    color="warning",   # Gelbe Farbe
                    size="sm",         # Klein
                    outline=True,      # Nur Umrandung, nicht gefüllt
                    className="theme-toggle-btn"
                ),
                # Dark Mode Button
                dbc.Button(
                    [html.I(className="fas fa-moon me-1"), "Dark"],  # Mond-Icon + Text
                    id="btn-dark-mode",
                    color="secondary",  # Graue Farbe
                    size="sm",
                    outline=True,
                    className="theme-toggle-btn"
                ),
            ], className="float-end")  # Nach rechts ausrichten
        ], width=4, className="text-end"),
    ], className="my-3 align-items-center"),  # my-3 = Margin Y (oben/unten)
    
    html.Div(id="theme-output", style={"display": "none"}),
    
    create_market_ticker(),
    
    dbc.Tabs([
        dbc.Tab(label="Portfolio", children=[
            # Sub-Tabs innerhalb des Portfolio-Tabs
            dbc.Tabs([
                dbc.Tab(label="Übersicht", children=[
                    dbc.Row([
                        dbc.Col([
                            # Button-Gruppe für Portfolio-Aktionen
                            dbc.ButtonGroup([
                                # Button zum Öffnen des Kauf/Verkauf-Modals
                                dbc.Button("💰 Buy/Sell", id="btn-buy-sell", color="primary", size="sm"),
                                # Button zum Öffnen der Transaktionshistorie
                                dbc.Button("📋 Transactions", id="btn-transactions", color="success", size="sm"),
                                # Button zum Öffnen des Kontostand-Modals
                                dbc.Button("💵 Kontostand", id="btn-kontostand", color="info", size="sm"),
                            ], className="mb-3"),
                        ], width=12),
                    ]),
                    html.Div(id="portfolio-table"),
                    html.Div(id="portfolio-summary", className="mt-3"),
                    # Kreisdiagramm der Portfolio-Zusammensetzung
                    dbc.Row([
                        dbc.Col([dcc.Graph(id="portfolio-chart", style={"height": "400px"})], width=12),
                    ], className="mt-3"),
                ], className="p-3"),  # p-3 = Padding
                
                # Zeigt die zeitliche Entwicklung des Portfolio-Werts
                dbc.Tab(label="Wertentwicklung", children=[
                    # Einzelne Aktien als Kacheln oben
                    html.H6("📊 Einzelne Positionen", className="mb-3"),
                    html.Div(id="portfolio-stock-cards", className="mb-4"),  # Kacheln für jede Aktie
                    
                    html.Hr(),  # Horizontale Trennlinie
                    
                    # Gesamtportfolio unten
                    html.H6("📈 Gesamtportfolio-Entwicklung", className="mb-3"),
                    html.Div(id="portfolio-total-summary", className="mb-3"),  # Gesamtwert-Chart
                ], className="p-3"),
            ]),
        ], className="p-3"),
        
        # TAB 2: AKTIEN - Aktiensuche, Kurs-Charts und News
        # anzeigen und aktuelle Nachrichten lesen.
        dbc.Tab(label="Aktien", children=[
            dbc.Row([
                # Linke Spalte: Suchfeld
                dbc.Col([
                    # Eingabefeld für die Aktiensuche
                    # debounce=True: Wartet bis der Benutzer aufhört zu tippen
                    dbc.Input(
                        id="stock-search", 
                        placeholder="Aktie suchen (z.B. Apple, TSLA)...",  # Platzhalter-Text
                        type="text", 
                        debounce=True  # Verzögert Callback bis Eingabe beendet
                    ),
                ], width=6),
                
                # Rechte Spalte: Zeitraum-Buttons
                dbc.Col([
                    # Button-Gruppe für verschiedene Zeiträume
                    dbc.ButtonGroup([
                        dbc.Button("1T", id="btn-1d", size="sm", outline=True, color="primary"),   # 1 Tag
                        dbc.Button("1W", id="btn-1w", size="sm", outline=True, color="primary"),   # 1 Woche
                        dbc.Button("1M", id="btn-1m", size="sm", outline=True, color="primary", active=True),  # 1 Monat (Standard)
                        dbc.Button("3M", id="btn-3m", size="sm", outline=True, color="primary"),   # 3 Monate
                        dbc.Button("1J", id="btn-1y", size="sm", outline=True, color="primary"),   # 1 Jahr
                        dbc.Button("Max", id="btn-max", size="sm", outline=True, color="primary"), # Maximum
                    ]),
                ], width=6),
            ], className="mb-3"),
            
            # Untere Zeile: Chart und News nebeneinander
            dbc.Row([
                # Linke Spalte: Kurs-Chart (8 von 12 Spalten breit)
                dbc.Col([dcc.Graph(id="stock-chart", style={"height": "400px"})], width=8),
                
                # Rechte Spalte: News-Liste (4 von 12 Spalten breit)
                dbc.Col([
                    html.H6("📰 News"),
                    # Scrollbarer Container für News
                    html.Div(id="stock-news", style={"maxHeight": "380px", "overflowY": "auto"})
                ], width=4),
            ]),
        ], className="p-3"),
        
        # TAB 3: NEWS - Aktuelle Finanznachrichten
        # in einem schönen Kachel-Layout mit Bildern.
        dbc.Tab(label="📰 News", children=[
            html.Div([
                dbc.Row([
                    # Überschrift und Untertext
                    dbc.Col([
                        html.H4("📰 Finanznachrichten", className="mb-0 text-white"),
                        html.Small("Aktuelle Nachrichten aus der Finanzwelt", className="text-muted")
                    ], width=12, lg=4),  # lg=4 bedeutet: auf großen Bildschirmen 4 Spalten
                    
                    # Suchfeld für Nachrichten
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("🔍"),  # Lupe-Icon
                            dbc.Input(
                                id="news-search-input",
                                placeholder="Suche nach Aktien, Themen...",
                                type="text",
                                debounce=True  # Wartet bis Eingabe beendet
                            ),
                        ], size="sm")
                    ], width=12, lg=5, className="mt-2 mt-lg-0"),  # mt-lg-0: kein Margin-Top auf großen Screens
                    
                    # Aktualisieren-Button
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("🔄 Aktualisieren", id="btn-refresh-news", color="primary", size="sm"),
                        ], className="float-end")
                    ], width=12, lg=3, className="mt-2 mt-lg-0 text-end"),
                ], className="mb-4 align-items-center"),
                
                # Unter-Tabs zum Filtern nach Kategorie
                dbc.Tabs([
                    dbc.Tab(label="🌍 Alle", tab_id="news-all"),       # Alle Nachrichten
                    dbc.Tab(label="📈 Aktien", tab_id="news-stocks"),  # Nur Aktien-News
                    dbc.Tab(label="₿ Krypto", tab_id="news-crypto"),    # Nur Krypto-News
                    dbc.Tab(label="🏦 Wirtschaft", tab_id="news-economy"),  # Wirtschaftsnews
                ], id="news-category-tabs", active_tab="news-all", className="mb-4"),
                
                dbc.Spinner(
                    html.Div(id="market-news", className="news-grid-container"),
                    color="primary",
                    type="border",
                    size="lg"  # Großer Spinner
                ),
            ], style={"minHeight": "80vh"})  # Mindestens 80% der Bildschirmhöhe
        ], className="p-3 bg-dark"),  # bg-dark = dunkler Hintergrund
        
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
                                    {"label": "Alle", "value": "all"},
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
                
                dbc.Tab(label="Prognose", children=[
                    dbc.Tabs([
                        # ARIMA Sub-Sub-Tab
                        dbc.Tab(label="📈 ARIMA", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H6("🔮 ARIMA Kursprognose"),
                                    html.P("Zeitreihen-basierte Kursprognose mit dem ARIMA-Modell. Berücksichtigt historische Trends.", className="text-muted"),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.InputGroupText("🔍"),
                                        dbc.Input(id="forecast-search-input", placeholder="Aktie suchen (z.B. Apple, Tesla)...", type="text", debounce=True),
                                    ], className="mb-2"),
                                    dbc.Select(
                                        id="forecast-stock-dropdown",
                                        options=[],
                                        placeholder="Bitte zuerst eine Aktie suchen...",
                                        className="mb-3"
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Small("Historische Daten", className="text-muted"),
                                    dbc.Select(
                                        id="forecast-history-select",
                                        options=[
                                            {"label": "3 Monate", "value": "3mo"},
                                            {"label": "6 Monate", "value": "6mo"},
                                            {"label": "1 Jahr", "value": "1y"},
                                            {"label": "2 Jahre", "value": "2y"},
                                            {"label": "5 Jahre", "value": "5y"},
                                        ],
                                        value="1y",
                                        className="mb-3"
                                    ),
                                ], width=2),
                                dbc.Col([
                                    html.Small("Prognose-Horizont", className="text-muted"),
                                    dbc.Select(
                                        id="forecast-days-select",
                                        options=[
                                            {"label": "1 Woche", "value": "7"},
                                            {"label": "2 Wochen", "value": "14"},
                                            {"label": "1 Monat", "value": "30"},
                                            {"label": "2 Monate", "value": "60"},
                                            {"label": "3 Monate", "value": "90"},
                                            {"label": "6 Monate", "value": "180"},
                                            {"label": "1 Jahr", "value": "365"},
                                            {"label": "2 Jahre", "value": "730"},
                                            {"label": "3 Jahre", "value": "1095"},
                                            {"label": "5 Jahre", "value": "1825"},
                                        ],
                                        value="30",
                                        className="mb-3"
                                    ),
                                ], width=2),
                                dbc.Col([
                                    html.Small(" ", className="d-block"),
                                    dbc.Button("🔮 Prognose erstellen", id="btn-forecast-analyze", color="success", className="w-100"),
                                ], width=2),
                                dbc.Col([
                                    dbc.Alert([
                                        html.Strong("Hinweis: "),
                                        "Für zuverlässige Prognosen sollte der historische Zeitraum mindestens so lang sein wie der Prognose-Horizont."
                                    ], color="info", className="mb-0 py-2 small"),
                                ], width=2),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        id="forecast-loading",
                                        type="circle",
                                        children=[
                                            html.Div(id="ai-forecast-output", className="mt-3"),
                                        ]
                                    ),
                                ], width=12),
                            ]),
                        ], className="p-3"),
                        
                        # Monte-Carlo Sub-Sub-Tab
                        dbc.Tab(label="🎲 Monte-Carlo", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H6("🎲 Monte-Carlo Simulation"),
                                    html.P("Stochastische Kursprognose basierend auf Geometric Brownian Motion mit tausenden Simulationspfaden.", className="text-muted"),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.InputGroupText("🔍"),
                                        dbc.Input(id="mc-search-input", placeholder="Aktie suchen (z.B. Apple, Tesla)...", type="text", debounce=True),
                                    ], className="mb-2"),
                                    dbc.Select(
                                        id="mc-stock-dropdown",
                                        options=[],
                                        placeholder="Bitte zuerst eine Aktie suchen...",
                                        className="mb-3"
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Small("Historische Daten", className="text-muted"),
                                    dbc.Select(
                                        id="mc-history-select",
                                        options=[
                                            {"label": "3 Monate", "value": "3mo"},
                                            {"label": "6 Monate", "value": "6mo"},
                                            {"label": "1 Jahr", "value": "1y"},
                                            {"label": "2 Jahre", "value": "2y"},
                                            {"label": "5 Jahre", "value": "5y"},
                                        ],
                                        value="1y",
                                        className="mb-3"
                                    ),
                                ], width=2),
                                dbc.Col([
                                    html.Small("Prognose-Horizont", className="text-muted"),
                                    dbc.Select(
                                        id="mc-days-select",
                                        options=[
                                            {"label": "1 Woche", "value": "7"},
                                            {"label": "2 Wochen", "value": "14"},
                                            {"label": "1 Monat", "value": "30"},
                                            {"label": "3 Monate", "value": "90"},
                                            {"label": "6 Monate", "value": "180"},
                                            {"label": "1 Jahr", "value": "365"},
                                            {"label": "2 Jahre", "value": "730"},
                                            {"label": "5 Jahre", "value": "1825"},
                                        ],
                                        value="30",
                                        className="mb-3"
                                    ),
                                ], width=2),
                                dbc.Col([
                                    html.Small("Simulationen", className="text-muted"),
                                    dbc.Select(
                                        id="mc-simulations-select",
                                        options=[
                                            {"label": "500", "value": "500"},
                                            {"label": "1.000", "value": "1000"},
                                            {"label": "5.000", "value": "5000"},
                                            {"label": "10.000", "value": "10000"},
                                        ],
                                        value="1000",
                                        className="mb-3"
                                    ),
                                ], width=2),
                                dbc.Col([
                                    html.Small(" ", className="d-block"),
                                    dbc.Button("🎲 Simulation starten", id="btn-mc-analyze", color="warning", className="w-100"),
                                ], width=2),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        id="mc-loading",
                                        type="circle",
                                        children=[
                                            html.Div(id="mc-output", className="mt-3"),
                                        ]
                                    ),
                                ], width=12),
                            ]),
                        ], className="p-3"),
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
                            html.Small("News-Anzahl", className="text-muted"),
                            dbc.Select(
                                id="corr-news-count",
                                options=[
                                    {"label": "100 News", "value": "100"},
                                    {"label": "200 News", "value": "200"},
                                    {"label": "500 News", "value": "500"},
                                    {"label": "1000 News", "value": "1000"},
                                    {"label": "Alle", "value": "all"},
                                ],
                                value="500",
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
            html.Div(id="buy-chart-container"),  # Chart wird nur bei Auswahl angezeigt
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
    
], fluid=True)  # fluid=True = Container nimmt volle Breite ein

# CALLBACKS - Die Interaktionslogik der Anwendung
# Diagrammen oder Texten).
#
# WICHTIGE KONZEPTE:
#
#
#
#    - Wie Input, aber löst den Callback NICHT aus
#    - Nützlich um zusätzliche Werte zu lesen
#
#
#
# Beispiel:
# @callback(
# )
# def meine_funktion(n_clicks):
#     return f"Button wurde {n_clicks} mal geklickt"

# CALLBACK: MARKT-TICKER AKTUALISIEREN
@callback(
    [Output(f"ticker-{s['name']}", "children") for s in MARKET_OVERVIEW_SYMBOLS] +
    [Output(f"ticker-{s['name']}", "style") for s in MARKET_OVERVIEW_SYMBOLS],
    
    Input("market-interval", "n_intervals")
)
def update_market_tickers(n):
    Parameter:
    - n: Anzahl der vergangenen Intervalle (wird nicht direkt verwendet,
         löst aber den Callback aus)
    
    Rückgabe: Liste von HTML-Elementen (Texte) + Liste von Style-Dictionaries
    """
    texts = []   # Liste für die anzuzeigenden Texte
    styles = []  # Liste für die Styling-Informationen
    
    # Durchlaufe alle Symbole aus der Konfiguration
    for s in MARKET_OVERVIEW_SYMBOLS:
        # Hole aktuellen Preis und Vortagesschluss
        price, prev = fetch_price(s["symbol"])
        
        if s.get("invert") and price:
            price = 1 / price
            if prev:
                prev = 1 / prev
        
        # Wenn kein Preis verfügbar: "n/a" anzeigen
        if price is None:
            texts.append(html.Span([html.B(s["name"]), ": n/a"]))
            styles.append({"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa", "padding": "8px"})
        else:
            decimals = s.get("decimals", 2)
            # Deutsche Zahlenformatierung: 1.234,56 statt 1,234.56
            formatted = f"{price:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            # Farbe basierend auf Änderung zum Vortag
            if prev:
                diff = price - prev
                color = "#22c55e" if diff > 0.0001 else "#ef4444" if diff < -0.0001 else "#000000"
            else:
                color = "#000000"
            
            texts.append(html.Span([html.B(s["name"]), f": {formatted}"], style={"color": color, "fontWeight": "bold"}))
            styles.append({"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa", "padding": "8px"})
    
    return texts + styles

# CALLBACK: AKTIENSUCHE UND CHART-AKTUALISIERUNG
# Dieser Callback reagiert auf:
# 1. Texteingabe im Suchfeld
# 2. Klicks auf die Zeitraum-Buttons (1T, 1W, 1M, etc.)
@callback(
    Output("stock-chart", "figure"),   # Aktualisiert das Diagramm
    Output("stock-news", "children"),   # Aktualisiert die News-Liste
    Output("btn-1d", "active"),         # Aktiv-Status 1-Tag Button
    Output("btn-1w", "active"),         # Aktiv-Status 1-Woche Button
    Output("btn-1m", "active"),         # Aktiv-Status 1-Monat Button
    Output("btn-3m", "active"),         # Aktiv-Status 3-Monate Button
    Output("btn-1y", "active"),         # Aktiv-Status 1-Jahr Button
    Output("btn-max", "active"),        # Aktiv-Status Maximum Button
    Input("stock-search", "value"),     # Suchfeld-Eingabe
    Input("btn-1d", "n_clicks"),        # 1-Tag Button
    Input("btn-1w", "n_clicks"),        # 1-Woche Button
    Input("btn-1m", "n_clicks"),        # 1-Monat Button
    Input("btn-3m", "n_clicks"),        # 3-Monate Button
    Input("btn-1y", "n_clicks"),        # 1-Jahr Button
    Input("btn-max", "n_clicks"),       # Maximum Button
    prevent_initial_call=True            # Nicht beim Laden ausführen
)
def update_stock_view(search, n1d, n1w, n1m, n3m, n1y, nmax):
    
    Parameter:
    - search: Der Suchbegriff aus dem Eingabefeld
    - n1d bis nmax: Klick-Zähler der Zeitraum-Buttons (Werte werden nicht direkt
                    verwendet, aber der Klick löst den Callback aus)
    
    Rückgabe: (chart_figure, news_elements, btn_1d_active, btn_1w_active, btn_1m_active, btn_3m_active, btn_1y_active, btn_max_active)
    """
    # Finde heraus, welches Element den Callback ausgelöst hat
    triggered = ctx.triggered_id
    
    # Mapping: Button-ID -> (Periode, Interval)
    period_map = {
        "btn-1d": ("1d", "5m"),      # 1 Tag, alle 5 Minuten
        "btn-1w": ("5d", "15m"),     # 5 Tage, alle 15 Minuten
        "btn-1m": ("1mo", "1d"),     # 1 Monat, täglich
        "btn-3m": ("3mo", "1d"),     # 3 Monate, täglich
        "btn-1y": ("1y", "1wk"),     # 1 Jahr, wöchentlich
        "btn-max": ("max", "1mo"),   # Maximum, monatlich
    }
    
    # Bestimme welcher Button aktiv sein soll
    active_btn = triggered if triggered in period_map else "btn-1m"
    btn_states = [active_btn == btn for btn in ["btn-1d", "btn-1w", "btn-1m", "btn-3m", "btn-1y", "btn-max"]]
    
    # Hole Periode und Interval basierend auf geklicktem Button
    # Falls keiner geklickt: Standard ist 1 Monat
    period, interval = period_map.get(triggered, ("1mo", "1d"))
    
    # Prüfe ob Suchbegriff lang genug ist
    if not search or len(search) < 2:
        return go.Figure(), html.P("Bitte Aktie suchen..."), *btn_states
    
    # Suche nach passenden Aktien
    results = search_stocks(search)
    if not results:
        return go.Figure(), html.P("Keine Ergebnisse"), *btn_states
    
    # Nimm das erste Suchergebnis
    symbol = results[0]["symbol"]
    
    # Erstelle den Chart für diese Aktie
    fig = create_stock_chart(symbol, period, interval)
    
    # Hole News zur Aktie (maximal 10)
    news = fetch_google_news(symbol, 10)
    
    # Erstelle News-Karten oder zeige "Keine News"
    news_items = [
        dbc.Card([
            dbc.CardBody([
                # Klickbarer Link zur News
                html.A(n["title"], href=n["link"], target="_blank", className="text-decoration-none"),
                # Quelle der News
                html.Small(f" — {n['source']}", className="text-muted d-block")
            ], className="p-2")
        ], className="mb-2") for n in news
    ] if news else [html.P("Keine News gefunden")]
    
    return fig, news_items, *btn_states

# CALLBACK: PORTFOLIO ANZEIGE AKTUALISIEREN
# Daten im portfolio-store ändern (z.B. nach Kauf/Verkauf).
@callback(
    Output("portfolio-table", "children"),      # Die Portfolio-Tabelle
    Output("portfolio-summary", "children"),    # Die Zusammenfassung (Investiert/Wert/P&L)
    Output("portfolio-chart", "figure"),        # Das Kreisdiagramm
    Output("portfolio-stock-cards", "children"), # Die einzelnen Aktien-Kacheln
    Output("portfolio-total-summary", "children"), # Gesamtportfolio-Zusammenfassung
    Input("portfolio-store", "data")            # Wird ausgelöst wenn sich Store ändert
)
def update_portfolio(portfolio):
    Parameter:
    - portfolio: Liste der Portfolio-Positionen aus dem Store
    
    Rückgabe: (tabelle, zusammenfassung, kreisdiagramm, aktienkacheln, gesamtsummary)
    """
    
    # Hilfsfunktion für leere/Fehler-Charts
    def empty_figure(text="Portfolio ist leer"):
        fig.update_layout(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    if not portfolio:
        return (
            html.P("Portfolio ist leer. Nutze Buy/Sell um Aktien hinzuzufügen.", className="text-muted"), 
            "", 
            empty_figure(), 
            html.P("Keine Positionen vorhanden", className="text-muted"),
            ""
        )
    
    try:
        rows = []
        stock_cards = []
        total_invested = 0
        total_value = 0
        
        for item in portfolio:
            symbol = item["symbol"]
            name = fetch_name(symbol) or symbol
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
                
                # Kachel für diese Aktie erstellen
                is_profit = pnl >= 0
                card_color = "success" if is_profit else "danger"
                card_bg = "linear-gradient(135deg, #1a472a 0%, #0d2818 100%)" if is_profit else "linear-gradient(135deg, #4a1a1a 0%, #2d0f0f 100%)"
                
                stock_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6(symbol, className="mb-0 text-white fw-bold"),
                                        html.Small(name[:20] + "..." if len(name) > 20 else name, className="text-muted"),
                                    ], width=8),
                                    dbc.Col([
                                        html.Span(f"{qty}x", className="badge bg-secondary")
                                    ], width=4, className="text-end"),
                                ], className="mb-2"),
                                html.Hr(className="my-2", style={"borderColor": "rgba(255,255,255,0.2)"}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Small("Kaufkurs", className="text-muted d-block"),
                                        html.Span(f"${buy_price:.2f}", className="text-warning"),
                                    ], width=6),
                                    dbc.Col([
                                        html.Small("Aktuell", className="text-muted d-block"),
                                        html.Span(f"${current_price:.2f}", className="text-info"),
                                    ], width=6),
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Small("Investiert", className="text-muted d-block"),
                                        html.Span(f"${invested:.2f}", className="text-white"),
                                    ], width=6),
                                    dbc.Col([
                                        html.Small("Wert", className="text-muted d-block"),
                                        html.Span(f"${value:.2f}", className="text-white"),
                                    ], width=6),
                                ], className="mb-2"),
                                html.Hr(className="my-2", style={"borderColor": "rgba(255,255,255,0.2)"}),
                                dbc.Row([
                                    dbc.Col([
                                        html.H5(
                                            f"{'+'if pnl >= 0 else ''}{pnl:.2f}$",
                                            className=f"mb-0 text-{card_color}"
                                        ),
                                        html.Small(
                                            f"({'+'if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)",
                                            className=f"text-{card_color}"
                                        ),
                                    ], width=12, className="text-center"),
                                ]),
                            ], className="p-3")
                        ], style={"background": card_bg, "border": f"1px solid {'#22c55e' if is_profit else '#ef4444'}", "borderRadius": "12px"})
                    ], xs=12, sm=6, md=4, lg=3, className="mb-3")
                )
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
        
        # Kacheln in einer Row
        cards_row = dbc.Row(stock_cards) if stock_cards else html.P("Keine Kursdaten verfügbar", className="text-muted")
        
        table = dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in ["Symbol", "Name", "Anzahl", "Kaufkurs", "Aktuell", "Investiert", "Wert", "P/L"]],
            style_cell={
                "textAlign": "center", 
                "padding": "12px",
                "backgroundColor": "#303030",
                "color": "#ffffff",
                "border": "1px solid #444"
            },
            style_header={
                "fontWeight": "bold", 
                "backgroundColor": "#404040",
                "color": "#ffffff",
                "border": "1px solid #555"
            },
            style_table={
                "borderRadius": "8px",
                "overflow": "hidden"
            },
            style_data_conditional=[
                {"if": {"filter_query": "{P/L} contains '+'", "column_id": "P/L"}, "color": "#22c55e", "fontWeight": "bold"},
                {"if": {"filter_query": "{P/L} contains '-'", "column_id": "P/L"}, "color": "#ef4444", "fontWeight": "bold"},
                {"if": {"state": "active"}, "backgroundColor": "#505050", "border": "1px solid #666"},
                {"if": {"state": "selected"}, "backgroundColor": "#505050", "border": "1px solid #666"},
            ],
            style_as_list_view=False,
            css=[{"selector": ".dash-table-tooltip", "rule": "background-color: #303030; color: white"}]
        )
        
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        summary = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("💰 Investiert", className="text-muted mb-1"),
                        html.H4(f"${total_invested:,.2f}", className="text-info mb-0")
                    ], width=4, className="text-center"),
                    dbc.Col([
                        html.H6("📊 Aktueller Wert", className="text-muted mb-1"),
                        html.H4(f"${total_value:,.2f}", className="text-primary mb-0")
                    ], width=4, className="text-center"),
                    dbc.Col([
                        html.H6("📈 Gewinn/Verlust", className="text-muted mb-1"),
                        html.H4(
                            f"${total_pnl:+,.2f}",
                            className=f"text-{'success' if total_pnl >= 0 else 'danger'} mb-0"
                        ),
                        html.Small(
                            f"({total_pnl_pct:+.2f}%)",
                            className=f"text-{'success' if total_pnl >= 0 else 'danger'}"
                        )
                    ], width=4, className="text-center"),
                ])
            ])
        ], className="mt-3 border-0", style={"background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"})
        
        # Charts mit Fehlerbehandlung erstellen
        try:
            chart = create_portfolio_pie_chart(portfolio)
        except Exception:
            chart = empty_figure("Fehler beim Laden des Pie-Charts")
        
        try:
            value_chart = create_portfolio_value_chart(portfolio)
        except Exception as e:
            value_chart = empty_figure(f"Fehler: {str(e)[:40]}")
        
        # Gesamtportfolio Summary für den Wertentwicklung Tab
        total_summary = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4("💰 Investiert", className="text-muted mb-2"),
                        html.H2(f"${total_invested:,.2f}", className="text-warning mb-0")
                    ], width=4, className="text-center"),
                    dbc.Col([
                        html.H4("📊 Aktueller Wert", className="text-muted mb-2"),
                        html.H2(f"${total_value:,.2f}", className="text-info mb-0")
                    ], width=4, className="text-center"),
                    dbc.Col([
                        html.H4("📈 Gewinn/Verlust", className="text-muted mb-2"),
                        html.H2(
                            f"${total_pnl:+,.2f}",
                            className=f"text-{'success' if total_pnl >= 0 else 'danger'} mb-0"
                        ),
                        html.H5(
                            f"({total_pnl_pct:+.2f}%)",
                            className=f"text-{'success' if total_pnl >= 0 else 'danger'}"
                        )
                    ], width=4, className="text-center"),
                ])
            ])
        ], className="border-0", style={"background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)", "borderRadius": "12px"})
        
        return table, summary, chart, cards_row, total_summary
        
    except Exception as e:
        # Fallback bei allgemeinem Fehler
        error_msg = html.Div([
            html.P(f"Fehler beim Laden des Portfolios: {str(e)}", className="text-danger"),
            html.P("Bitte versuchen Sie es später erneut.", className="text-muted")
        ])
        return error_msg, "", empty_figure(), "", ""

# Market News
@callback(
    Output("market-news", "children"),
    Input("btn-refresh-news", "n_clicks"),
    Input("news-category-tabs", "active_tab"),
    Input("news-search-input", "value"),
    prevent_initial_call=False
)
def update_market_news(n, category, search_term):
    # Kategorien definieren
    category_targets = {
        "news-all": ["DAX", "Nasdaq", "S&P 500", "Bitcoin", "Gold", "Tesla", "Apple", "Microsoft"],
        "news-stocks": ["DAX", "Nasdaq", "S&P 500", "Tesla", "Apple", "Microsoft", "Amazon", "Google"],
        "news-crypto": ["Bitcoin", "Ethereum", "Crypto", "Binance", "Solana"],
        "news-economy": ["Economy", "Federal Reserve", "Inflation", "Interest Rate", "GDP"],
    }
    
    # Suchbegriff oder Kategorie verwenden
    if search_term and len(search_term) >= 2:
        targets = [search_term]
        news_limit = 20
    else:
        targets = category_targets.get(category, category_targets["news-all"])
        news_limit = 4
    
    all_news = []
    for target in targets:
        news = fetch_google_news(target, news_limit)
        all_news.extend(news)
    
    if not all_news:
        return html.Div([
            html.Div([
                html.I(className="fas fa-newspaper fa-4x text-muted mb-3"),
                html.H5("Keine Nachrichten gefunden", className="text-muted"),
                html.P("Versuchen Sie einen anderen Suchbegriff oder eine andere Kategorie.", className="text-muted small")
            ], className="text-center py-5")
        ])
    
    all_finance_images = [
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1518546305927-5a555bb7020d?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1622630998477-20aa696ecb05?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1617788138017-80ad40651399?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1611186871348-b1ce696e52c9?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1633419461186-7d40a38105ec?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1523474253046-8cd2748b5fd2?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1573804633927-bfcbcd909acd?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1610375461246-83df859d849d?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1541354329998-f4d9a9f9297f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1642790106117-e829e14a795f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1642790551116-18e150f248e3?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1642543492481-44e81e3914a7?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1535320903710-d993d3d77d29?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1543286386-713bdd548da4?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1565514020179-026b92b2d9b3?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1444653614773-995cb1ef9efa?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1462206092226-f46025ffe607?w=400&h=200&fit=crop",
    ]
    
    used_images = set()
    link_to_image = {}  # Gleiche Webpage = gleiches Bild
    
    def get_unique_image(news_link, index):
        """
        # Basis-Domain extrahieren für Duplikat-Erkennung
        try:
            from urllib.parse import urlparse
            domain = urlparse(news_link).netloc
        except:
            domain = news_link
        
        # Wenn diese Domain schon ein Bild hat, verwende dasselbe
        if domain in link_to_image:
            return link_to_image[domain]
        
        # Finde ein noch nicht verwendetes Bild
        for img_idx, img_url in enumerate(all_finance_images):
            if img_url not in used_images:
                used_images.add(img_url)
                link_to_image[domain] = img_url
                return img_url
        
        fallback_img = all_finance_images[index % len(all_finance_images)]
        link_to_image[domain] = fallback_img
        return fallback_img
    
    # Kategorie-Farben
    def get_category_color(symbol):
        symbol_lower = symbol.lower()
        if any(x in symbol_lower for x in ["bitcoin", "ethereum", "crypto", "binance", "solana"]):
            return "warning"  # Gold für Krypto
        elif any(x in symbol_lower for x in ["dax", "nasdaq", "s&p"]):
            return "primary"  # Blau für Indizes
        elif any(x in symbol_lower for x in ["economy", "federal", "inflation", "interest", "gdp"]):
            return "info"  # Cyan für Wirtschaft
        elif any(x in symbol_lower for x in ["gold", "silver", "oil"]):
            return "secondary"  # Grau für Rohstoffe
        return "success"  # Grün für Aktien
    
    # Datum formatieren
    def format_date(pub_date):
        if not pub_date:
            return "Gerade eben"
        try:
            from datetime import datetime
            # Parse RSS date format: "Wed, 18 Dec 2024 10:30:00 GMT"
            dt = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
            now = datetime.now()
            diff = now - dt
            if diff.days > 0:
                return f"vor {diff.days} Tag{'en' if diff.days > 1 else ''}"
            hours = diff.seconds // 3600
            if hours > 0:
                return f"vor {hours} Stunde{'n' if hours > 1 else ''}"
            minutes = diff.seconds // 60
            return f"vor {minutes} Minute{'n' if minutes != 1 else ''}"
        except:
            return pub_date[:16] if len(pub_date) > 16 else pub_date
    
    # News-Kacheln erstellen
    news_cards = []
    for i, news_item in enumerate(all_news[:24]):  # Max 24 News
        symbol = news_item.get("symbol", "")
        news_link = news_item.get("link", "")
        card = dbc.Col([
            dbc.Card([
                # Vorschaubild
                html.Div([
                    html.Img(
                        src=get_unique_image(news_link, i),
                        style={
                            "width": "100%",
                            "height": "140px",
                            "objectFit": "cover",
                            "borderTopLeftRadius": "0.375rem",
                            "borderTopRightRadius": "0.375rem"
                        }
                    ),
                    # Kategorie-Badge auf dem Bild
                    dbc.Badge(
                        symbol,
                        color=get_category_color(symbol),
                        className="position-absolute",
                        style={"top": "10px", "left": "10px", "fontSize": "0.7rem"}
                    ),
                ], style={"position": "relative"}),
                
                # Card Body
                dbc.CardBody([
                    html.H6(
                        html.A(
                            news_item["title"][:80] + ("..." if len(news_item["title"]) > 80 else ""),
                            href=news_item["link"],
                            target="_blank",
                            className="text-decoration-none text-white stretched-link",
                            style={"fontSize": "0.9rem", "lineHeight": "1.3"}
                        ),
                        className="card-title mb-2",
                        style={"minHeight": "45px"}
                    ),
                    html.Div([
                        html.Small([
                            html.I(className="fas fa-newspaper me-1"),
                            html.Span(news_item.get("source", "Unbekannt")[:20], className="text-muted"),
                        ], className="d-block"),
                        html.Small([
                            html.I(className="fas fa-clock me-1"),
                            html.Span(format_date(news_item.get("pubDate", "")), className="text-muted"),
                        ]),
                    ], className="mt-auto")
                ], className="d-flex flex-column", style={"minHeight": "120px"})
            ], className="h-100 bg-dark border-secondary news-card", style={
                "transition": "transform 0.2s, box-shadow 0.2s",
                "cursor": "pointer"
            })
        ], xs=12, sm=6, md=4, lg=3, className="mb-4")
        news_cards.append(card)
    
    return dbc.Row(news_cards, className="g-3")

# CALLBACK: BUY/SELL MODAL ÖFFNEN/SCHLIESSEN
@callback(
    Output("buy-sell-modal", "is_open"),    # Steuert ob Modal offen ist (True/False)
    Input("btn-buy-sell", "n_clicks"),       # "Buy/Sell" Button im Portfolio
    Input("btn-close-modal", "n_clicks"),    # "Schließen" Button im Modal
    Input("btn-confirm-buy", "n_clicks"),    # "Kaufen" Button (schließt auch)
    Input("btn-confirm-sell", "n_clicks"),   # "Verkaufen" Button (schließt auch)
    State("buy-sell-modal", "is_open"),      # Aktueller Zustand des Modals
    prevent_initial_call=True
)
def toggle_buy_sell_modal(n1, n2, n3, n4, is_open):
    Parameter:
    - n1-n4: Klick-Zähler der verschiedenen Buttons
    - is_open: Aktueller Zustand (True = offen, False = geschlossen)
    
    Rückgabe: Der neue Zustand (invertiert)
    """
    return not is_open  # Einfach umschalten: offen -> zu, zu -> offen

# CALLBACK: KONTOSTAND MODAL ÖFFNEN/SCHLIESSEN
@callback(
    Output("kontostand-modal", "is_open"),
    Input("btn-kontostand", "n_clicks"),     # "Kontostand" Button
    Input("btn-close-kontostand", "n_clicks"), # "Schließen" Button
    State("kontostand-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_kontostand_modal(n1, n2, is_open):

# Dieser Callback hat mehrere Funktionen:
# 1. Kontostand anzeigen wenn Modal geöffnet wird
# 2. Einzahlung verarbeiten
# 3. Auszahlung verarbeiten
@callback(
    Output("kontostand-display", "children"),  # Anzeige des aktuellen Kontostands
    Output("balance-message", "children"),      # Erfolgs-/Fehlermeldung
    Input("kontostand-modal", "is_open"),       # Modal wurde geöffnet/geschlossen
    Input("btn-deposit", "n_clicks"),           # Einzahlung-Button
    Input("btn-withdraw", "n_clicks"),          # Auszahlung-Button
    State("balance-amount", "value"),           # Eingegebener Betrag
    prevent_initial_call=True
)
def handle_kontostand(is_open, btn_deposit, btn_withdraw, amount):
    """
    Sucht nach Aktien für den Kauf und erstellt klickbare Buttons.
    
    Parameter:
    - query: Der Suchbegriff aus dem Eingabefeld
    
    Rückgabe: (buttons_liste, suchergebnisse_daten)
        return [], []
    
    # Suche durchführen
    results = search_stocks(query)
    
    # Erstelle Buttons für jedes Ergebnis (maximal 5)
    buttons = [
        dbc.Button(
            f"{r['symbol']} - {r['name']}",  # Button-Text
            id={"type": "search-result", "index": i},  # Dynamische ID
            color="light", 
            className="w-100 mb-1 text-start",  # Volle Breite, kleiner Abstand
            size="sm"
        )
        for i, r in enumerate(results[:5])  # Nur erste 5 Ergebnisse
    ]
    
    return buttons, results[:5]  # Rückgabe: Buttons und Daten

# (Balance displayed/updated by calculate_total)

# CALLBACK: AKTIE FÜR KAUF/VERKAUF AUSWÄHLEN
# Das ALL bedeutet: Reagiere auf ALLE Elemente dieses Typs
@callback(
    Output("buy-stock-info", "children"),    # Aktien-Info Anzeige
    Output("buy-chart-container", "children"), # Mini-Chart im Modal
    Output("selected-ticker", "data"),       # Speichert ausgewählte Aktie
    Input({"type": "search-result", "index": dash.ALL}, "n_clicks"),  # Pattern Match auf alle Buttons
    State("search-results-store", "data"),   # Gespeicherte Suchergebnisse
    prevent_initial_call=True
)
def select_stock_for_buy(clicks, results):
    """
    Führt den Kauf einer Aktie durch.
    
    Schritte:
    1. Validierung der Eingaben
    2. Prüfen ob genügend Geld vorhanden ist
    3. Portfolio aktualisieren (neue Position oder vorhandene erweitern)
    4. Transaktion speichern
    5. Kontostand aktualisieren
    
    Parameter:
    - n: Klick-Zähler des Buttons
    - ticker: Daten der ausgewählten Aktie {symbol, name, price}
    - qty: Gewünschte Kaufmenge
    - portfolio: Aktuelles Portfolio
    
    Rückgabe: Das aktualisierte Portfolio
        return portfolio or []

    # Sicherheitsprüfung: Genug Geld vorhanden?
    balance = load_balance()
    total_cost = int(qty) * float(ticker.get("price", 0))

    if total_cost > balance:
        return portfolio or []
    
    portfolio = portfolio or []  # Falls None, leere Liste verwenden
    
    # Prüfen ob diese Aktie schon im Portfolio ist
    found = False
    for item in portfolio:
        if item["symbol"] == ticker["symbol"]:
            old_qty = item["qty"]
            old_price = item.get("buy_price") or item.get("avg_price", 0)
            new_qty = old_qty + int(qty)
            new_price = ((old_price * old_qty) + (ticker["price"] * int(qty))) / new_qty
            
            # Werte aktualisieren
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
    
    # Portfolio in Datei speichern
    save_portfolio(portfolio)
    
    # Transaktion in Historie speichern
    save_transaction({
        "timestamp": datetime.now().isoformat(),  # Aktuelles Datum/Zeit im ISO-Format
        "type": "buy",                            # Transaktionstyp
        "symbol": ticker["symbol"],
        "qty": int(qty),
        "price": ticker["price"]
    })
    
    # Kaufpreis vom Kontostand abziehen
    try:
        current_balance = load_balance()
        cost = int(qty) * float(ticker.get("price", 0))
        current_balance -= cost
        save_balance(current_balance)
    except Exception:
        pass

    return portfolio

# CALLBACK: VERKAUF BESTÄTIGEN
@callback(
    Output("portfolio-store", "data", allow_duplicate=True),
    Input("btn-confirm-sell", "n_clicks"),    # Verkaufen-Button
    State("selected-ticker", "data"),         # Ausgewählte Aktie
    State("buy-qty", "value"),                # Zu verkaufende Menge
    State("portfolio-store", "data"),         # Aktuelles Portfolio
    prevent_initial_call=True
)
def confirm_sell(n, ticker, qty, portfolio):
    """
    Verwaltet das Transaktions-Historie-Modal.
    
    Args:
        n1: Klick-Zähler für "Transaktionen"-Button
        n2: Klick-Zähler für Schließen-Button
        year: Ausgewähltes Jahr zum Filtern (oder "all")
        month: Ausgewählter Monat zum Filtern (oder "all")
        tx_type: Transaktionstyp zum Filtern ("buy", "sell" oder "all")
        is_open: Aktueller Modal-Zustand
    
    Returns:
        Tuple: (Modal-Status, Tabellen-Inhalt, Zusammenfassung, Jahr-Optionen)
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
        style_cell={
            "textAlign": "center", 
            "padding": "10px",
            "backgroundColor": "#303030",
            "color": "#ffffff",
            "border": "1px solid #444"
        },
        style_header={
            "fontWeight": "bold", 
            "backgroundColor": "#404040",
            "color": "#ffffff",
            "border": "1px solid #555"
        },
        style_table={
            "borderRadius": "8px",
            "overflow": "hidden",
            "maxHeight": "400px",
            "overflowY": "auto"
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Typ} = 'Kauf'"}, "backgroundColor": "#1a472a", "color": "#22c55e"},
            {"if": {"filter_query": "{Typ} = 'Verkauf'"}, "backgroundColor": "#4a1a1a", "color": "#ef4444"},
        ],
        page_size=50,  # Zeigt bis zu 50 Transaktionen
        page_action="native",
        sort_action="native",
        sort_mode="single",
        filter_action="native"
    )
    
    saldo = total_sell - total_buy
    summary = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("📊 Transaktionen", className="text-muted mb-1"),
                    html.H4(f"{len(filtered)}", className="text-info mb-0")
                ], width=3, className="text-center"),
                dbc.Col([
                    html.H6("💵 Käufe", className="text-muted mb-1"),
                    html.H4(f"${total_buy:,.2f}", className="text-success mb-0")
                ], width=3, className="text-center"),
                dbc.Col([
                    html.H6("💸 Verkäufe", className="text-muted mb-1"),
                    html.H4(f"${total_sell:,.2f}", className="text-danger mb-0")
                ], width=3, className="text-center"),
                dbc.Col([
                    html.H6("📈 Saldo", className="text-muted mb-1"),
                    html.H4(f"${saldo:+,.2f}", className=f"text-{'success' if saldo >= 0 else 'danger'} mb-0")
                ], width=3, className="text-center"),
            ])
        ])
    ], className="mt-3 border-0", style={"background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"})
    
    return is_open, table, summary, year_options

# TICKER DETAIL MODAL - Market Overview Detailansicht
# öffnet sich dieses Modal mit detaillierten Informationen.
#
# Features:
# - Zeigt Detailchart für den ausgewählten Index/ETF
# - Aktuelle Statistiken (Kurs, Hoch, Tief, Volumen)
#
# Dynamische Inputs:
# - Das ermöglicht Klick-Handler für jeden einzelnen Ticker
#
# *args Pattern:
# - Der Code muss dann die einzelnen Argumente extrahieren
@callback(
    # --- Outputs ---
    Output("ticker-modal", "is_open"),               # Modal öffnen/schließen
    Output("ticker-modal-header", "children"),       # Titel im Modal-Header
    Output("ticker-modal-stats", "children"),        # Statistik-Anzeige
    Output("ticker-modal-chart", "figure"),          # Chart-Figur
    Output("current-ticker-symbol", "data"),         # Speichert aktuelles Symbol
    Output("ticker-btn-1d", "active"),               # Aktiv-Status 1-Tag Button
    Output("ticker-btn-1w", "active"),               # Aktiv-Status 1-Woche Button
    Output("ticker-btn-1m", "active"),               # Aktiv-Status 1-Monat Button
    Output("ticker-btn-3m", "active"),               # Aktiv-Status 3-Monate Button
    
    [Input(f"ticker-{s['name']}", "n_clicks") for s in MARKET_OVERVIEW_SYMBOLS] +
    [Input("btn-close-ticker", "n_clicks"),
     Input("ticker-btn-1d", "n_clicks"),             # 1-Tages-Ansicht
     Input("ticker-btn-1w", "n_clicks"),             # 1-Wochen-Ansicht
     Input("ticker-btn-1m", "n_clicks"),             # 1-Monats-Ansicht
     Input("ticker-btn-3m", "n_clicks")],            # 3-Monats-Ansicht
    
    # --- States ---
    State("ticker-modal", "is_open"),
    State("current-ticker-symbol", "data"),          # Gespeichertes Symbol für Zeitraum-Wechsel
    prevent_initial_call=True
)
def toggle_ticker_modal(*args):
    """
    Sucht Aktien für die Sentiment-Analyse basierend auf der Benutzereingabe.
    
    Args:
        search_query: Der Suchbegriff (Aktienname oder Symbol)
    
    Returns:
        Tuple: (Dropdown-Optionen, vorausgewählter Wert)
    
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

# SENTIMENT ANALYSE - Hauptanalyse-Callback
#
# Technische Details:
#
# Die Analyse zeigt:
# - Durchschnittlicher Sentiment-Score
# - Anzahl analysierter News
# - Kursänderung im Zeitraum
# - Chart mit Sentiment über Zeit
# - Top News sortiert nach Sentiment-Stärke
@callback(
    Output("ai-sentiment-output", "children"),       # Ergebnis-Container
    Input("btn-sentiment-analyze", "n_clicks"),      # "Analysieren"-Button
    State("sentiment-stock-dropdown", "value"),      # Ausgewählte Aktie
    State("sentiment-period-select", "value"),       # Zeitraum (1 Monat, 3 Monate, etc.)
    State("sentiment-news-count", "value"),          # Anzahl News zu analysieren
    prevent_initial_call=True
)
def sentiment_analyze_callback(n_clicks, symbol, period, news_count):
    """
    Sucht Aktien für die Korrelationsanalyse.
    
    results = search_stocks(search_query)
    if not results:
        return [{"label": "Keine Ergebnisse gefunden", "value": "", "disabled": True}], None
    
    options = [
        {"label": f"{r['name']} ({r['symbol']}) - {r['exchange']}", "value": r['symbol']}
        for r in results
    ]
    
    default_value = results[0]['symbol'] if len(results) == 1 else None
    
    return options, default_value

# KORRELATIONS-ANALYSE - Hauptanalyse-Callback
# und Kursbewegung einer Aktie.
#
# Was ist Korrelation?
# - Werte von -1 bis +1:
#   * 0 = Keine lineare Beziehung
#
# Die Analyse zeigt:
# - Korrelationskoeffizient zwischen Sentiment und Kurs
# - Visualisierung der Beziehung im Chart
# - Interpretation der Ergebnisse
@callback(
    Output("corr-output", "children"),               # Ergebnis-Container
    Input("btn-corr-analyze", "n_clicks"),           # "Analysieren"-Button
    State("corr-stock-dropdown", "value"),           # Ausgewählte Aktie
    State("corr-period-select", "value"),            # Zeitraum
    State("corr-news-count", "value"),               # Anzahl News
    prevent_initial_call=True
)
def correlation_analyze_callback(n_clicks, symbol, period, news_count):
    """
    Sucht Aktien für die Prognose-Funktion.
    
    Verwendet Yahoo Finance API direkt statt der globalen search_stocks(),
    um die Implementierung zu demonstrieren.
    
    options = []
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={search_term}&quotesCount=10&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        
        for quote in data.get("quotes", []):
            symbol = quote.get("symbol", "")
            name = quote.get("shortname", "") or quote.get("longname", "")
            qtype = quote.get("quoteType", "")
            
            if qtype in ["EQUITY", "ETF"] and symbol:
                label = f"{symbol} - {name}" if name else symbol
                options.append({"label": label, "value": symbol})
    except:
        pass
    
    default_value = options[0]["value"] if options else None
    return options, default_value

# --- ARIMA-Prognose Hauptanalyse ---
@callback(
    Output("ai-forecast-output", "children"),        # Ergebnis-Container
    Input("btn-forecast-analyze", "n_clicks"),       # "Analysieren"-Button
    State("forecast-stock-dropdown", "value"),       # Ausgewählte Aktie
    State("forecast-history-select", "value"),       # Historische Daten (z.B. "1y" für 1 Jahr)
    State("forecast-days-select", "value"),          # Prognose-Horizont in Tagen
    prevent_initial_call=True
)
def forecast_analyze_callback(n_clicks, symbol, history_period, forecast_days):
    """
    Sucht Aktien für die Monte-Carlo-Simulation.
    
    options = []
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={search_term}&quotesCount=10&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        
        for quote in data.get("quotes", []):
            symbol = quote.get("symbol", "")
            name = quote.get("shortname", "") or quote.get("longname", "")
            qtype = quote.get("quoteType", "")
            
            if qtype in ["EQUITY", "ETF"] and symbol:
                label = f"{symbol} - {name}" if name else symbol
                options.append({"label": label, "value": symbol})
    except:
        pass
    
    default_value = options[0]["value"] if options else None
    return options, default_value

# --- Monte-Carlo Hauptanalyse ---
# Dieser Callback führt die Monte-Carlo-Simulation durch
@callback(
    Output("mc-output", "children"),                 # Ergebnis-Container
    Input("btn-mc-analyze", "n_clicks"),             # "Analysieren"-Button
    State("mc-stock-dropdown", "value"),             # Ausgewählte Aktie
    State("mc-history-select", "value"),             # Historische Daten für Volatilität
    State("mc-days-select", "value"),                # Prognose-Horizont
    State("mc-simulations-select", "value"),         # Anzahl Simulationen
    prevent_initial_call=True
)
def monte_carlo_analyze_callback(n_clicks, symbol, history_period, forecast_days, num_simulations):
    """
    function(n_light, n_dark) {
        // dash_clientside.callback_context enthält Infos darüber,
        // welches Element den Callback ausgelöst hat
        const triggered = dash_clientside.callback_context.triggered[0];
        if (!triggered) return window.dash_clientside.no_update;
        
        // Extrahiere die ID des auslösenden Elements
        const triggeredId = triggered.prop_id.split('.')[0];
        
        // Je nach Button: Theme-Klasse hinzufügen oder entfernen
        if (triggeredId === 'btn-light-mode') {
            document.body.classList.add('light-mode');
            return 'light';  // Speichere im Store
        } else if (triggeredId === 'btn-dark-mode') {
            document.body.classList.remove('light-mode');
            return 'dark';   // Speichere im Store
        }
        return window.dash_clientside.no_update;
    }
    # Inputs: Die Theme-Toggle-Buttons
    Input("btn-light-mode", "n_clicks"),
    Input("btn-dark-mode", "n_clicks"),
    prevent_initial_call=True
)

# SERVER STARTEN - Hier startet die Anwendung
# (nicht wenn es von einem anderen Skript importiert wird).
#
# __name__ == "__main__" ist ein Python-Idiom:
# - Wenn du es importierst: __name__ ist der Modulname
#
# app.run() startet den Dash-Entwicklungsserver:
#

if __name__ == "__main__":
    # Begrüßungsnachricht in der Konsole
    print("=" * 50)
    print("🚀 Stock Dashboard startet...")
    print("📊 Öffne im Browser: http://localhost:8050")
    print("=" * 50)
    
    # Server starten
    # debug=True ermöglicht:
    # - Automatisches Neuladen bei Codeänderungen
    # - Detaillierte Fehlermeldungen im Browser
    # - Hot Reloading
    app.run(debug=True, port=8050)
