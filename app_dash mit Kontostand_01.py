import dash
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table,no_update
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
from flask import Flask, request, session, redirect

# Sentiment-Analyse Modul importieren
from sentiment_analysis import (
    VADER_AVAILABLE,
    analyze_sentiment,
    analyze_correlation,
    get_sentiment_label,
    get_correlation_label,
    ARIMA_AVAILABLE,
    analyze_forecast,
    get_forecast_label,
    analyze_monte_carlo,
    get_monte_carlo_label,
)

# ----------------- Logindaten-Pfade (optional) -----------------
USERS_FILE = Path(__file__).parent / "user_data.json"
USERS = json.loads(USERS_FILE.read_text(encoding="utf-8"))

# ----------------- Flask-Server (für Login/Session) -----------------
server = Flask(__name__)
server.secret_key = "uni-projekt-secret-key"  # für Session nötig


# ----------------- Login (minimal) -----------------
@server.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u in USERS and USERS[u]["password"] == p:
            session["logged_in"] = True
            session["username"] = u
            session["role"] = USERS[u]["role"]   # "user" oder "admin"

            # OPTIONAL: "Angemeldet bleiben" (Session bleibt länger bestehen)
            # session.permanent = bool(request.form.get("remember"))

            return redirect("/")
        error = "Falscher Benutzername oder Passwort."

    return f"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Login</title>

  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    body {{
      min-height: 100vh;
      background:
        radial-gradient(1200px 600px at 15% 10%, rgba(13,110,253,.18), transparent 60%),
        radial-gradient(900px 500px at 85% 25%, rgba(25,135,84,.12), transparent 55%),
        linear-gradient(180deg, #f8fafc, #eef2ff);
    }}
    .login-card {{
      border: 1px solid rgba(15, 23, 42, .08);
      border-radius: 18px;
      box-shadow: 0 18px 55px rgba(0,0,0,.10);
      overflow: hidden;
    }}
    .brand {{
      display:flex;
      align-items:center;
      gap: 12px;
    }}
    .brand img {{
      width: 44px;
      height: 44px;
      object-fit: contain;
    }}
    .muted {{
      color: #64748b;
    }}
    .form-control {{
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .btn-primary {{
      border-radius: 12px;
      padding: 12px 14px;
      font-weight: 600;
    }}
    .footer {{
      font-size: .85rem;
      color: #64748b;
      text-align:center;
    }}
    .footer a {{
      color: #64748b;
      text-decoration: none;
    }}
    .footer a:hover {{
      color: #0d6efd;
      text-decoration: underline;
    }}
  </style>
</head>

<body>
  <div class="container">
    <div class="row vh-100 align-items-center justify-content-center">
      <div class="col-12 col-md-7 col-lg-5 col-xl-4">

        <div class="login-card bg-white">
          <div class="p-4 p-sm-5">
            <div class="brand mb-3 justify-content-center">
              <img src="/assets/logo.png" alt="Logo">
              <div class="text-center">
                <h4 class="mb-0">Anmelden</h4>
                <div class="small muted">Kontostand-Dashboard</div>
              </div>
            </div>

            {f'<div class="alert alert-danger rounded-3">{error}</div>' if error else ''}

            <div class="small muted text-center mb-3">
              Bitte melde dich mit deinen Zugangsdaten an.
            </div>

            <form method="post">
              <div class="mb-3">
                <label class="form-label">Benutzername</label>
                <input class="form-control" name="username" autocomplete="username" required>
              </div>

              <div class="mb-3">
                <label class="form-label">Passwort</label>
                <div class="input-group">
                  <input class="form-control" id="pw" type="password" name="password" autocomplete="current-password" required>
                  <button class="btn btn-outline-secondary" type="button" id="togglePw" style="border-radius:12px;">
                    <i class="bi bi-eye"></i>
                  </button>
                </div>
              </div>

              <div class="d-flex justify-content-between align-items-center mb-3">
                <div class="form-check">
                  <input class="form-check-input" type="checkbox" name="remember" value="1" id="remember">
                  <label class="form-check-label small muted" for="remember">Angemeldet bleiben</label>
                </div>
                <a class="small muted" href="#" style="text-decoration:none;">Passwort vergessen?</a>
              </div>

              <button class="btn btn-primary w-100" type="submit">Anmelden</button>
            </form>

            <div class="footer mt-4">
              © <span id="year"></span> FH Südwestfalen ·
              <a href="#">Impressum</a> · <a href="#">Datenschutz</a> · <a href="#">Kontakt</a>
              <div class="mt-1">Version 0.9</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    document.getElementById("year").textContent = new Date().getFullYear();
    const btn = document.getElementById("togglePw");
    const pw  = document.getElementById("pw");
    btn.addEventListener("click", () => {{
      const isPw = pw.type === "password";
      pw.type = isPw ? "text" : "password";
      btn.innerHTML = isPw ? '<i class="bi bi-eye-slash"></i>' : '<i class="bi bi-eye"></i>';
    }});
  </script>
</body>
</html>
    """


@server.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@server.before_request
def protect():
    allowed = (
        "/login", "/logout",
        "/_dash", "/_dash-component-suites",
        "/assets", "/favicon.ico"
    )
    if request.path.startswith(allowed):
        return None
    if not session.get("logged_in"):
        return redirect("/login")
    return None


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
                # Zufällige Farben oder feste Palette
                colors.append(f"hsl({hash(symbol) % 360}, 70%, 50%)")
        
        if not values:
            fig = go.Figure()
            fig.add_annotation(text="Keine aktuellen Preise verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            insidetextorientation='radial',
            textfont=dict(color="white")
        )])
        
        fig.update_layout(
            title=dict(text="Portfolio-Zusammensetzung", font=dict(color="white")),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Fehler beim Laden", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

def create_portfolio_value_chart(portfolio):
    """Erstellt ein Liniendiagramm mit Kauflinie, grün/rot Bereichen und Gewinn/Verlust-Anzeige."""
    
    def empty_fig(text):
        fig = go.Figure()
        fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    if not portfolio:
        return empty_fig("Portfolio ist leer")
    
    try:
        # Daten sammeln
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
        
        if not data_list:
            return empty_fig("Keine aktuellen Kursdaten verfügbar")
        
        # Sortieren nach Symbol
        data_list.sort(key=lambda x: x["symbol"])
        
        symbols = [d["name"][:15] if d["name"] else d["symbol"] for d in data_list]
        invested_values = [d["invested"] for d in data_list]
        current_values = [d["current"] for d in data_list]
        pnl_values = [d["pnl"] for d in data_list]
        
        fig = go.Figure()
        
        # Füllbereich: Grün über Kauflinie, Rot unter Kauflinie
        for i, d in enumerate(data_list):
            x_pos = [i - 0.3, i + 0.3, i + 0.3, i - 0.3]
            if d["current"] >= d["invested"]:
                # Grüner Bereich (Gewinn)
                y_fill = [d["invested"], d["invested"], d["current"], d["current"]]
                fig.add_trace(go.Scatter(
                    x=x_pos, y=y_fill,
                    fill="toself",
                    fillcolor="rgba(34, 197, 94, 0.3)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            else:
                # Roter Bereich (Verlust)
                y_fill = [d["current"], d["current"], d["invested"], d["invested"]]
                fig.add_trace(go.Scatter(
                    x=x_pos, y=y_fill,
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.3)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
        # Kauflinie (gestrichelt)
        fig.add_trace(go.Scatter(
            x=list(range(len(symbols))),
            y=invested_values,
            mode="lines+markers",
            name="Kaufwert",
            line=dict(color="#fbbf24", width=3, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
            hovertemplate="<b>Kaufwert</b><br>%{y:,.2f} USD<extra></extra>"
        ))
        
        # Aktueller Wert (durchgezogen)
        colors = ["#22c55e" if pnl >= 0 else "#ef4444" for pnl in pnl_values]
        fig.add_trace(go.Scatter(
            x=list(range(len(symbols))),
            y=current_values,
            mode="lines+markers",
            name="Aktueller Wert",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
            hovertemplate="<b>Aktueller Wert</b><br>%{y:,.2f} USD<extra></extra>"
        ))
        
        # Gewinn/Verlust Annotationen an der Seite
        annotations = []
        for i, d in enumerate(data_list):
            color = "#22c55e" if d["pnl"] >= 0 else "#ef4444"
            sign = "+" if d["pnl"] >= 0 else ""
            annotations.append(dict(
                x=i,
                y=max(d["current"], d["invested"]) * 1.08,
                text=f"<b>{sign}{d['pnl']:.0f}$</b><br><span style='font-size:10px'>({sign}{d['pnl_pct']:.1f}%)</span>",
                showarrow=False,
                font=dict(color=color, size=11),
                align="center"
            ))
        
        # Gesamt P/L oben rechts
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        total_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        total_sign = "+" if total_pnl >= 0 else ""
        
        fig.update_layout(
            title=dict(
                text=f"Portfolio-Wertentwicklung | Gesamt: <span style='color:{total_color}'>{total_sign}{total_pnl:,.2f}$ ({total_sign}{total_pnl_pct:.1f}%)</span>",
                font=dict(size=14, color="white")
            ),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(symbols))),
                ticktext=symbols,
                tickfont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            ),
            yaxis=dict(
                title="Wert (USD)",
                tickformat=",.0f",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            ),
            annotations=annotations,
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


def create_portfolio_total_value_chart(portfolio, days=30):
    """Erstellt eine Zeitreihe des Gesamtwerts des Portfolios über die letzten `days` Tage.
    Für jede Position werden historische Schlusskurse abgefragt und mit der Menge multipliziert.
    """
    import datetime

    def empty_fig(text):
        fig = go.Figure()
        fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    if not portfolio:
        return empty_fig("Keine Positionen im Portfolio")

    try:
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
                    try:
                        dt = datetime.date.fromtimestamp(idx.timestamp())
                    except Exception:
                        continue

                if dt < start or dt > end:
                    continue

                try:
                    close = row.get("Close") if isinstance(row, dict) or hasattr(row, "get") else row["Close"]
                    value = float(close) * qty
                except Exception:
                    continue

                totals.setdefault(dt, 0.0)
                totals[dt] += value

        if not totals:
            return empty_fig("Keine historischen Preise verfügbar")

        dates = sorted(totals.keys())
        values = [totals[d] for d in dates]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name="Gesamtwert", line=dict(color="#0ea5a4")))

        # Gesamtinvestition als Referenz (aktueller investierter Betrag)
        total_invested = sum((item.get("qty", 0) * (item.get("buy_price") or item.get("avg_price", 0))) for item in portfolio)
        fig.add_trace(go.Scatter(x=dates, y=[total_invested] * len(dates), mode="lines", name="Investiert (aktuell)", line=dict(color="#2563eb", dash="dash")))

        fig.update_layout(
            title=dict(text=f"Gesamtwert des Portfolios (letzte {days} Tage)", font=dict(color="white")),
            xaxis_title="Datum",
            yaxis_title="Wert (USD)",
            hovermode="x unified",
            legend=dict(font=dict(color="white")),
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,30,30,0.5)",
            xaxis=dict(tickfont=dict(color="white"), titlefont=dict(color="white"), gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(tickfont=dict(color="white"), titlefont=dict(color="white"), gridcolor="rgba(255,255,255,0.1)")
        )

        return fig
    except Exception:
        return empty_fig("Fehler beim Laden der Daten")

# ============== App Initialisierung ==============
app = dash.Dash(
    __name__, 
    server=server, 
    external_stylesheets=[
        dbc.themes.DARKLY,  # Dark Theme für bessere News-Darstellung
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"  # FontAwesome Icons
    ], 
    suppress_callback_exceptions=True
)
app.title = "Stock Dashboard"

# Custom CSS für News-Kacheln Hover-Effekte und Theme-Support
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* News Card Styles */
            .news-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 123, 255, 0.3) !important;
            }
            .news-grid-container {
                min-height: 60vh;
            }
            .news-card .card-body {
                background: linear-gradient(180deg, #2d3436 0%, #1e272e 100%);
            }
            .news-card img {
                filter: brightness(0.9);
                transition: filter 0.3s;
            }
            .news-card:hover img {
                filter: brightness(1.1);
            }
            
            /* Light Mode Overrides */
            body.light-mode {
                background-color: #f8f9fa !important;
                color: #212529 !important;
            }
            body.light-mode .bg-dark {
                background-color: #ffffff !important;
            }
            body.light-mode .text-white {
                color: #212529 !important;
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
            
            /* Portfolio Table Styles */
            .portfolio-table th {
                font-weight: 600;
            }
            .portfolio-table td {
                vertical-align: middle;
            }
            
            /* Theme Toggle Button */
            .theme-toggle-btn {
                transition: all 0.3s ease;
            }
            .theme-toggle-btn:hover {
                transform: scale(1.1);
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

# ============== Layout ==============
def create_market_ticker():
    return dbc.Row([
        dbc.Col(html.Div(id=f"ticker-{s['name']}", className="text-center p-2", 
                         style={"cursor": "pointer", "borderRadius": "5px", "background": "rgba(128, 128, 128, 0.2)"}), 
                width="auto") 
        for s in MARKET_OVERVIEW_SYMBOLS
    ], className="g-2 p-2 mb-3", justify="center", style={"background": "rgba(128, 128, 128, 0.1)", "borderRadius": "8px"})

app.layout = dbc.Container([
    dcc.Interval(id="market-interval", interval=15000, n_intervals=0),
    dcc.Store(id="selected-ticker", data=None),
    dcc.Store(id="portfolio-store", data=load_portfolio()),
    dcc.Store(id="search-results-store", data=[]),
    dcc.Store(id="theme-store", data="dark"),  # Theme Store
    
    # Header mit Theme Toggle
    dbc.Row([
        dbc.Col([
            html.H4("📈 Stock Dashboard", className="mb-0"),
        ], width=8),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="fas fa-sun me-1"), "Light"],
                    id="btn-light-mode",
                    color="warning",
                    size="sm",
                    outline=True,
                    className="theme-toggle-btn"
                ),
                dbc.Button(
                    [html.I(className="fas fa-moon me-1"), "Dark"],
                    id="btn-dark-mode",
                    color="secondary",
                    size="sm",
                    outline=True,
                    className="theme-toggle-btn"
                ),
            ], className="float-end")
        ], width=4, className="text-end"),
    ], className="my-3 align-items-center"),
    
    # Dummy output für Theme-Toggle (JavaScript wird über clientside callback gesteuert)
    html.Div(id="theme-output", style={"display": "none"}),
    
    # Market Ticker Bar
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
                                dbc.Button("📋 Transactions", id="btn-transactions", color="success", size="sm"),
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
                    # Einzelne Aktien als Kacheln oben
                    html.H6("📊 Einzelne Positionen", className="mb-3"),
                    html.Div(id="portfolio-stock-cards", className="mb-4"),
                    
                    html.Hr(),
                    
                    # Gesamtportfolio unten
                    html.H6("📈 Gesamtportfolio-Entwicklung", className="mb-3"),
                    html.Div(id="portfolio-total-summary", className="mb-3"),
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
        dbc.Tab(label="📰 News", children=[
            html.Div([
                # Header mit Suchfeld und Aktualisieren-Button
                dbc.Row([
                    dbc.Col([
                        html.H4("📰 Finanznachrichten", className="mb-0 text-white"),
                        html.Small("Aktuelle Nachrichten aus der Finanzwelt", className="text-muted")
                    ], width=12, lg=4),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("🔍"),
                            dbc.Input(
                                id="news-search-input",
                                placeholder="Suche nach Aktien, Themen...",
                                type="text",
                                debounce=True
                            ),
                        ], size="sm")
                    ], width=12, lg=5, className="mt-2 mt-lg-0"),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("🔄 Aktualisieren", id="btn-refresh-news", color="primary", size="sm"),
                        ], className="float-end")
                    ], width=12, lg=3, className="mt-2 mt-lg-0 text-end"),
                ], className="mb-4 align-items-center"),
                
                # Kategorie-Tabs
                dbc.Tabs([
                    dbc.Tab(label="🌍 Alle", tab_id="news-all"),
                    dbc.Tab(label="📈 Aktien", tab_id="news-stocks"),
                    dbc.Tab(label="₿ Krypto", tab_id="news-crypto"),
                    dbc.Tab(label="🏦 Wirtschaft", tab_id="news-economy"),
                ], id="news-category-tabs", active_tab="news-all", className="mb-4"),
                
                # News-Kacheln Container
                dbc.Spinner(
                    html.Div(id="market-news", className="news-grid-container"),
                    color="primary",
                    type="border",
                    size="lg"
                ),
            ], style={"minHeight": "80vh"})
        ], className="p-3 bg-dark"),
        
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
                
                # Prognose Sub-Tab (mit ARIMA und Monte-Carlo als Unter-Tabs)
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
    Output("portfolio-stock-cards", "children"),
    Output("portfolio-total-summary", "children"),
    Input("portfolio-store", "data")
)
def update_portfolio(portfolio):
    # Leere Figur für Fehlerfälle
    def empty_figure(text="Portfolio ist leer"):
        fig = go.Figure()
        fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
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
    
    # Große Sammlung von Finanz-Bildern für einzigartige Vorschaubilder
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
    
    # Tracking für bereits verwendete Bilder und Link-Bild-Zuordnung
    used_images = set()
    link_to_image = {}  # Gleiche Webpage = gleiches Bild
    
    def get_unique_image(news_link, index):
        """Gibt ein einzigartiges Bild zurück, außer die Webpage wurde schon verwendet"""
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
        
        # Fallback: Wenn alle Bilder verwendet wurden, nehme eines basierend auf Index
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
    Output("buy-chart-container", "children"),
    Output("selected-ticker", "data"),
    Input({"type": "search-result", "index": dash.ALL}, "n_clicks"),
    State("search-results-store", "data"),
    prevent_initial_call=True
)
def select_stock_for_buy(clicks, results):
    if not any(clicks) or not results:
        return "", "", None
    
    idx = next((i for i, c in enumerate(clicks) if c), 0)
    if idx >= len(results):
        return "", "", None
    
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
    chart_container = dcc.Graph(figure=fig, style={"height": "250px"})
    return info, chart_container, {"symbol": symbol, "name": stock["name"], "price": price}

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
def sentiment_analyze_callback(n_clicks, symbol, period, news_count):
    if not symbol:
        return dbc.Alert("Bitte wählen Sie eine Aktie aus der Dropdown-Liste aus.", color="warning")
    
    if not VADER_AVAILABLE:
        return dbc.Alert("Die vaderSentiment-Bibliothek ist nicht installiert. Bitte installieren Sie sie mit: pip install vaderSentiment", color="danger")
    
    # News-Limit verarbeiten ("all" = sehr hohe Zahl)
    news_limit = 100000 if news_count == "all" else int(news_count or 100)
    
    # Analyse durchführen über das ausgelagerte Modul
    result = analyze_sentiment(symbol, period, news_limit)
    
    if "error" in result:
        return dbc.Alert(result["error"], color="danger")
    
    # Ergebnisse extrahieren
    stats = result["stats"]
    news_items = result["news_items"]
    sources_found = result["sources_found"]
    fig = result["figure"]
    
    sentiment_label, sentiment_color = get_sentiment_label(stats["avg_sentiment"])
    sign = "+" if stats["pct_change"] >= 0 else ""
    
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
    
    sources_info = ", ".join(sources_found) if sources_found else "Keine Quellen"
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Durchschnittlicher Sentiment", className="card-title"),
                        html.H2(f"{stats['avg_sentiment']:.2f}", className=f"text-{sentiment_color}"),
                        dbc.Badge(sentiment_label.upper(), color=sentiment_color, className="mt-2")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Analysierte News", className="card-title"),
                        html.H2(f"{stats['news_count']}", className="text-primary"),
                        html.Small("Artikel analysiert", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Kursänderung", className="card-title"),
                        html.H2(f"{sign}{stats['pct_change']:.2f}%", className=f"text-{'success' if stats['is_positive'] else 'danger'}"),
                        html.Small(f"{stats['start_price']:.2f} → {stats['end_price']:.2f} USD", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Sentiment-Tage", className="card-title"),
                        html.H2(f"{stats['sentiment_days']}", className="text-info"),
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
    State("corr-news-count", "value"),
    prevent_initial_call=True
)
def correlation_analyze_callback(n_clicks, symbol, period, news_count):
    if not symbol:
        return dbc.Alert("Bitte wählen Sie eine Aktie aus der Dropdown-Liste aus.", color="warning")
    
    if not VADER_AVAILABLE:
        return dbc.Alert("Die vaderSentiment-Bibliothek ist nicht installiert.", color="danger")
    
    # News-Limit verarbeiten
    news_limit = 100000 if news_count == "all" else int(news_count or 500)
    
    # Analyse durchführen über das ausgelagerte Modul
    result = analyze_correlation(symbol, period, news_limit)
    
    if "error" in result:
        return dbc.Alert(result["error"], color="warning")
    
    # Ergebnisse extrahieren
    stats = result["stats"]
    correlation = result["correlation"]
    fig = result["figure"]
    
    corr_label, corr_color = get_correlation_label(correlation)
    
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
                        html.H2(f"{stats['news_count']}", className="text-primary"),
                        html.Small(f"über {stats['days_back']} Tage", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Kursänderung", className="card-title"),
                        html.H2(f"{'+' if stats['pct_change'] >= 0 else ''}{stats['pct_change']:.2f}%", className=f"text-{'success' if stats['is_positive'] else 'danger'}"),
                        html.Small(f"{stats['start_price']:.2f} → {stats['end_price']:.2f}", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Ø Sentiment", className="card-title"),
                        html.H2(f"{stats['avg_sentiment']:.3f}", className=f"text-{'success' if stats['avg_sentiment'] > 0 else 'danger'}"),
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


# ============== Forecast Callbacks ==============
# Stock search for forecast
@callback(
    Output("forecast-stock-dropdown", "options"),
    Output("forecast-stock-dropdown", "value"),
    Input("forecast-search-input", "value"),
    prevent_initial_call=True
)
def forecast_search_callback(search_term):
    if not search_term or len(search_term) < 2:
        return [], None
    
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


# Forecast Analysis
@callback(
    Output("ai-forecast-output", "children"),
    Input("btn-forecast-analyze", "n_clicks"),
    State("forecast-stock-dropdown", "value"),
    State("forecast-history-select", "value"),
    State("forecast-days-select", "value"),
    prevent_initial_call=True
)
def forecast_analyze_callback(n_clicks, symbol, history_period, forecast_days):
    if not symbol:
        return dbc.Alert("Bitte wählen Sie eine Aktie aus der Dropdown-Liste aus.", color="warning")
    
    if not ARIMA_AVAILABLE:
        return dbc.Alert("Die statsmodels-Bibliothek ist nicht installiert. Bitte installieren Sie sie mit: pip install statsmodels", color="danger")
    
    forecast_days_int = int(forecast_days) if forecast_days else 30
    
    # Analyse durchführen
    result = analyze_forecast(symbol, history_period, forecast_days_int)
    
    if "error" in result:
        return dbc.Alert(result["error"], color="danger")
    
    # Ergebnisse extrahieren
    stats = result["stats"]
    fig = result["figure"]
    
    forecast_label, forecast_color = get_forecast_label(stats["forecast_change"])
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Aktueller Kurs", className="card-title"),
                        html.H2(f"${stats['current_price']:.2f}", className="text-primary"),
                        html.Small("Letzter Schlusskurs", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"Prognose ({stats['forecast_days']} Tage)", className="card-title"),
                        html.H2(f"${stats['forecast_price']:.2f}", className=f"text-{forecast_color}"),
                        dbc.Badge(forecast_label, color=forecast_color, className="mt-1")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Erwartete Änderung", className="card-title"),
                        html.H2(f"{'+' if stats['forecast_change'] >= 0 else ''}{stats['forecast_change']:.2f}%", className=f"text-{forecast_color}"),
                        html.Small(f"{stats['current_price']:.2f} → {stats['forecast_price']:.2f}", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("95% Konfidenz", className="card-title"),
                        html.H6(f"${stats['ci_lower']:.2f} - ${stats['ci_upper']:.2f}", className="text-info"),
                        html.Small(f"ARIMA{stats['arima_order']}", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
        ], className="mb-3"),
        dbc.Alert([
            html.Strong("⚠️ Hinweis: "),
            "Diese Prognose basiert auf einem statistischen ARIMA-Modell und dient nur zu Informationszwecken. ",
            "Aktienkurse werden von vielen Faktoren beeinflusst, die das Modell nicht berücksichtigt. ",
            html.Strong("Keine Anlageberatung!")
        ], color="warning", className="mb-3"),
        dcc.Graph(figure=fig),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("📊 Modell-Details", className="card-title"),
                        html.P([
                            html.Strong("ARIMA-Parameter: "), f"{stats['arima_order']}", html.Br(),
                            html.Strong("AIC-Score: "), f"{stats['aic']:.2f}", html.Br(),
                            html.Strong("Trainingsdaten: "), f"{stats['history_days']} Tage", html.Br(),
                            html.Strong("Prognose-Horizont: "), f"{stats['forecast_days']} Tage", html.Br(),
                            html.Strong("Hist. jährl. Rendite: "), f"{stats.get('annual_drift', 0):.1f}%", html.Br(),
                            html.Strong("Hist. jährl. Volatilität: "), f"{stats.get('annual_volatility', 0):.1f}%"
                        ], className="mb-0 small")
                    ])
                ])
            ], width=12),
        ]),
    ])


# ============== Monte-Carlo Callbacks ==============
# Stock search for Monte-Carlo
@callback(
    Output("mc-stock-dropdown", "options"),
    Output("mc-stock-dropdown", "value"),
    Input("mc-search-input", "value"),
    prevent_initial_call=True
)
def mc_search_callback(search_term):
    if not search_term or len(search_term) < 2:
        return [], None
    
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


# Monte-Carlo Analysis
@callback(
    Output("mc-output", "children"),
    Input("btn-mc-analyze", "n_clicks"),
    State("mc-stock-dropdown", "value"),
    State("mc-history-select", "value"),
    State("mc-days-select", "value"),
    State("mc-simulations-select", "value"),
    prevent_initial_call=True
)
def monte_carlo_analyze_callback(n_clicks, symbol, history_period, forecast_days, num_simulations):
    if not symbol:
        return dbc.Alert("Bitte wählen Sie eine Aktie aus der Dropdown-Liste aus.", color="warning")
    
    forecast_days_int = int(forecast_days) if forecast_days else 30
    num_simulations_int = int(num_simulations) if num_simulations else 1000
    
    # Analyse durchführen
    result = analyze_monte_carlo(symbol, history_period, forecast_days_int, num_simulations_int)
    
    if "error" in result:
        return dbc.Alert(result["error"], color="danger")
    
    # Ergebnisse extrahieren
    stats = result["stats"]
    fig = result["figure"]
    
    mc_label, mc_color = get_monte_carlo_label(stats["prob_positive"])
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Aktueller Kurs", className="card-title"),
                        html.H2(f"${stats['current_price']:.2f}", className="text-primary"),
                        html.Small("Letzter Schlusskurs", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"Median ({stats['forecast_days']}T)", className="card-title"),
                        html.H2(f"${stats['median_price']:.2f}", className=f"text-{mc_color}"),
                        dbc.Badge(mc_label, color=mc_color, className="mt-1")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Erwartete Änderung", className="card-title"),
                        html.H2(f"{'+' if stats['forecast_change'] >= 0 else ''}{stats['forecast_change']:.2f}%", className=f"text-{mc_color}"),
                        html.Small(f"Mittelwert: ${stats['mean_price']:.2f}", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("P(Gewinn)", className="card-title"),
                        html.H2(f"{stats['prob_positive']:.1f}%", className=f"text-{'success' if stats['prob_positive'] > 50 else 'danger'}"),
                        html.Small(f"{stats['num_simulations']:,} Simulationen", className="text-muted")
                    ])
                ], className="text-center")
            ], width=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("📊 Wahrscheinlichkeits-Szenarien", className="card-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Strong("P(+10%):", className="text-success"),
                                    html.Span(f" {stats['prob_up_10']:.1f}%")
                                ]),
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.Strong("P(-10%):", className="text-danger"),
                                    html.Span(f" {stats['prob_down_10']:.1f}%")
                                ]),
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.Strong("Volatilität (ann.):", className="text-info"),
                                    html.Span(f" {stats['sigma']*100:.1f}%")
                                ]),
                            ], width=4),
                        ]),
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("📈 Perzentile (Endpreis)", className="card-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Small("5%: ", className="text-muted"),
                                    html.Span(f"${stats['percentiles']['p5']:.2f}")
                                ]),
                                html.Div([
                                    html.Small("25%: ", className="text-muted"),
                                    html.Span(f"${stats['percentiles']['p25']:.2f}")
                                ]),
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.Small("50%: ", className="text-muted"),
                                    html.Strong(f"${stats['percentiles']['p50']:.2f}")
                                ]),
                                html.Div([
                                    html.Small("75%: ", className="text-muted"),
                                    html.Span(f"${stats['percentiles']['p75']:.2f}")
                                ]),
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.Small("95%: ", className="text-muted"),
                                    html.Span(f"${stats['percentiles']['p95']:.2f}")
                                ]),
                                html.Div([
                                    html.Small("Drift (ann.): ", className="text-muted"),
                                    html.Span(f"{stats['mu']*100:.1f}%")
                                ]),
                            ], width=4),
                        ]),
                    ])
                ])
            ], width=6),
        ], className="mb-3"),
        dbc.Alert([
            html.Strong("🎲 Monte-Carlo Simulation: "),
            f"Basierend auf {stats['num_simulations']:,} Simulationspfaden mit Geometric Brownian Motion. ",
            f"Die historische Volatilität beträgt {stats['sigma']*100:.1f}% (annualisiert). ",
            html.Strong("Die Simulation zeigt mögliche Kursverläufe, keine garantierten Ergebnisse!")
        ], color="info", className="mb-3"),
        dcc.Graph(figure=fig),
    ])


# ============== Theme Toggle Callback ==============
app.clientside_callback(
    """
    function(n_light, n_dark) {
        const triggered = dash_clientside.callback_context.triggered[0];
        if (!triggered) return window.dash_clientside.no_update;
        
        const triggeredId = triggered.prop_id.split('.')[0];
        
        if (triggeredId === 'btn-light-mode') {
            document.body.classList.add('light-mode');
            return 'light';
        } else if (triggeredId === 'btn-dark-mode') {
            document.body.classList.remove('light-mode');
            return 'dark';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-store", "data"),
    Input("btn-light-mode", "n_clicks"),
    Input("btn-dark-mode", "n_clicks"),
    prevent_initial_call=True
)


# ============== Server starten ==============
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Stock Dashboard startet...")
    print("📊 Öffne im Browser: http://localhost:8050")
    print("=" * 50)
    app.run(debug=True, port=8050)

