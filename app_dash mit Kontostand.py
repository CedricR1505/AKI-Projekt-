# ================================================================================
# STOCK DASHBOARD - Eine Web-Anwendung zur Aktienanalyse und Portfolio-Verwaltung
# ================================================================================
# Diese Anwendung ermöglicht:
# - Echtzeit-Aktienkurse anzeigen
# - Portfolio verwalten (Kaufen/Verkaufen)
# - Nachrichten zu Aktien lesen
# - KI-gestützte Analysen (Sentiment, Prognosen)
# ================================================================================

# ============== BIBLIOTHEKEN IMPORTIEREN ==============
# Hier laden wir alle notwendigen Bibliotheken (Module), die wir für unsere App brauchen

# DASH - Das Haupt-Framework für unsere Web-Anwendung
# Dash ist ein Python-Framework zum Erstellen von interaktiven Web-Dashboards
import dash

# Aus Dash importieren wir verschiedene Komponenten:
# - dcc (Dash Core Components): Interaktive Elemente wie Dropdowns, Graphen, Input-Felder
# - html: HTML-Elemente wie Überschriften, Absätze, Divs
# - Input/Output/State: Für die Reaktion auf Benutzeraktionen (Callbacks)
# - callback: Dekorator um Funktionen als Callback zu markieren
# - ctx: Context-Objekt um herauszufinden, welches Element geklickt wurde
# - dash_table: Zum Erstellen von interaktiven Tabellen
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table

# DASH BOOTSTRAP COMPONENTS - Schöne, vorgefertigte UI-Komponenten
# Bootstrap ist ein CSS-Framework für ansprechende Designs
import dash_bootstrap_components as dbc

# PLOTLY - Bibliothek zum Erstellen von interaktiven Diagrammen/Charts
# graph_objects gibt uns volle Kontrolle über die Diagramm-Erstellung
import plotly.graph_objects as go

# make_subplots erlaubt mehrere Diagramme in einem Bild
from plotly.subplots import make_subplots

# YFINANCE - Yahoo Finance API zum Abrufen von Aktiendaten
# Ermöglicht kostenlosen Zugriff auf Aktienkurse, historische Daten etc.
import yfinance as yf

# REQUESTS - Bibliothek für HTTP-Anfragen (z.B. API-Aufrufe, Webseiten abrufen)
import requests

# JSON - Zum Lesen und Schreiben von JSON-Dateien (ein Datenformat)
import json

# RE (Regular Expressions) - Zum Suchen von Mustern in Texten
# Wird hier für das Parsen von RSS-Feeds verwendet
import re

# PATHLIB - Modernes Modul für Dateipfad-Operationen
# Path macht das Arbeiten mit Dateien und Ordnern einfacher
from pathlib import Path

# DATETIME - Zum Arbeiten mit Datum und Uhrzeit
from datetime import datetime

# UNESCAPE - Zum Dekodieren von HTML-Entities (z.B. &amp; wird zu &)
from html import unescape

# PANDAS - Mächtige Bibliothek für Datenanalyse und -manipulation
# Wird oft für Tabellen und Zeitreihen verwendet
import pandas as pd

# ============== SENTIMENT-ANALYSE MODUL IMPORTIEREN ==============
# Hier importieren wir Funktionen aus unserer eigenen sentiment_analysis.py Datei
# Diese Funktionen führen KI-gestützte Analysen durch
from sentiment_analysis import (
    VADER_AVAILABLE,        # Boolean: Ist die VADER-Bibliothek installiert?
    analyze_sentiment,      # Funktion: Analysiert die Stimmung von Nachrichten
    analyze_correlation,    # Funktion: Berechnet Korrelation zwischen Sentiment und Kurs
    get_sentiment_label,    # Funktion: Gibt Label für Sentiment-Wert zurück (positiv/negativ)
    get_correlation_label,  # Funktion: Gibt Label für Korrelationswert zurück
    ARIMA_AVAILABLE,        # Boolean: Ist die ARIMA-Bibliothek installiert?
    analyze_forecast,       # Funktion: Erstellt Kursprognosen mit ARIMA-Modell
    get_forecast_label,     # Funktion: Gibt Label für Prognose zurück
    analyze_monte_carlo,    # Funktion: Führt Monte-Carlo Simulation durch
    get_monte_carlo_label,  # Funktion: Gibt Label für Monte-Carlo Ergebnis zurück
)

# ================================================================================
# KONFIGURATION: DATEIPFADE FÜR DIE DATENSPEICHERUNG
# ================================================================================
# Hier definieren wir, wo unsere Daten gespeichert werden sollen.
# Die App speichert Portfolio, Transaktionen und Kontostand in JSON-Dateien.

# DATA_DIR: Der Ordner, in dem alle Daten gespeichert werden
# Path(__file__) = Pfad zu dieser Python-Datei
# .parent = Der übergeordnete Ordner (der Ordner, in dem diese Datei liegt)
# / "gui" = Unterordner namens "gui" erstellen/verwenden
DATA_DIR = Path(__file__).parent / "gui"

# mkdir(exist_ok=True) = Erstelle den Ordner, falls er nicht existiert
# exist_ok=True verhindert einen Fehler, falls der Ordner schon da ist
DATA_DIR.mkdir(exist_ok=True)

# Hier definieren wir die Pfade zu unseren drei Datendateien:
# 1. PORTFOLIO_FILE: Speichert welche Aktien der Nutzer besitzt
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"

# 2. TRANSACTIONS_FILE: Speichert alle Käufe und Verkäufe (Historie)
TRANSACTIONS_FILE = DATA_DIR / "transactions.json"

# 3. BALANCE_FILE: Speichert den aktuellen Kontostand (virtuelles Geld)
BALANCE_FILE = DATA_DIR / "balance.json"

# ================================================================================
# KONFIGURATION: MARKTÜBERSICHT-SYMBOLE
# ================================================================================
# Diese Liste definiert die Finanzinstrumente, die oben in der Marktübersicht
# (dem "Ticker") angezeigt werden. Jedes Element ist ein Dictionary mit:
# - "name": Der Anzeigename (was der Nutzer sieht)
# - "symbol": Das Yahoo-Finance-Symbol zum Abrufen der Daten
# - "decimals": Anzahl der Nachkommastellen bei der Anzeige
# - "invert": Optional - wenn True, wird der Kehrwert angezeigt (für EUR/USD)

MARKET_OVERVIEW_SYMBOLS = [
    # Deutsche Aktienindizes
    {"name": "DAX", "symbol": "^GDAXI", "decimals": 0},      # DAX 40 - Die 40 größten deutschen Unternehmen
    {"name": "MDAX", "symbol": "^MDAXI", "decimals": 0},    # MDAX - Mittelgroße deutsche Unternehmen
    {"name": "SDAX", "symbol": "^SDAXI", "decimals": 0},    # SDAX - Kleinere deutsche Unternehmen
    
    # US-amerikanische Aktienindizes
    {"name": "Dow", "symbol": "^DJI", "decimals": 0},        # Dow Jones Industrial Average - 30 große US-Firmen
    {"name": "Nasdaq", "symbol": "^IXIC", "decimals": 0},   # Nasdaq Composite - Tech-lastig
    
    # Rohstoffe
    {"name": "Gold", "symbol": "GC=F", "decimals": 2},       # Goldpreis in USD pro Unze
    {"name": "Brent", "symbol": "BZ=F", "decimals": 2},      # Brent-Öl Preis in USD pro Barrel
    
    # Kryptowährung
    {"name": "BTC", "symbol": "BTC-USD", "decimals": 0},     # Bitcoin in US-Dollar
    
    # Währungspaar (mit invert=True wird aus USD/EUR -> EUR/USD)
    {"name": "EUR/USD", "symbol": "EURUSD=X", "decimals": 4, "invert": True},  # Euro zu US-Dollar Kurs
]

# ================================================================================
# HILFSFUNKTIONEN: DATENVERWALTUNG (Laden & Speichern)
# ================================================================================
# Diese Funktionen kümmern sich um das Speichern und Laden von Daten.
# Wir verwenden JSON-Dateien als einfache "Datenbank".
# ================================================================================

def load_portfolio():
    """
    Lädt das Portfolio (Liste aller gekauften Aktien) aus der JSON-Datei.
    
    Funktionsweise:
    1. Prüft, ob die Datei existiert
    2. Wenn ja: Liest den Inhalt und wandelt JSON in Python-Liste um
    3. Wenn nein oder Fehler: Gibt eine leere Liste zurück
    
    Rückgabe: Eine Liste von Dictionaries, z.B.:
    [{"symbol": "AAPL", "qty": 10, "buy_price": 150.0}, ...]
    """
    # Prüfe ob die Datei existiert
    if PORTFOLIO_FILE.exists():
        try:
            # Lese die Datei und parse den JSON-Inhalt
            # encoding="utf-8" stellt sicher, dass Sonderzeichen korrekt gelesen werden
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        except:
            # Bei Fehlern (z.B. ungültiges JSON) gebe leere Liste zurück
            return []
    # Wenn Datei nicht existiert, gebe leere Liste zurück
    return []


def save_portfolio(data):
    """
    Speichert das Portfolio in die JSON-Datei.
    
    Parameter:
    - data: Liste von Dictionaries mit den Portfolio-Positionen
    
    Die Funktion wandelt die Python-Liste in JSON-Format um und speichert sie.
    indent=2 macht die Datei menschenlesbar (schön formatiert).
    """
    PORTFOLIO_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_transactions():
    """
    Lädt alle Transaktionen (Käufe und Verkäufe) aus der JSON-Datei.
    
    Rückgabe: Eine Liste aller vergangenen Transaktionen, z.B.:
    [{"timestamp": "2024-01-15T10:30:00", "type": "buy", "symbol": "AAPL", 
      "qty": 5, "price": 150.0}, ...]
    """
    if TRANSACTIONS_FILE.exists():
        try:
            return json.loads(TRANSACTIONS_FILE.read_text(encoding="utf-8"))
        except:
            return []
    return []


def save_transaction(tx):
    """
    Fügt eine neue Transaktion zur Transaktionshistorie hinzu.
    
    Parameter:
    - tx: Dictionary mit Transaktionsdaten
          {"timestamp": "...", "type": "buy/sell", "symbol": "...", 
           "qty": Anzahl, "price": Preis}
    
    Die Funktion lädt erst alle bestehenden Transaktionen,
    fügt die neue hinzu und speichert dann alles.
    """
    # Lade bestehende Transaktionen
    txs = load_transactions()
    # Füge neue Transaktion hinzu
    txs.append(tx)
    # Speichere alle Transaktionen
    TRANSACTIONS_FILE.write_text(json.dumps(txs, indent=2), encoding="utf-8")


def load_balance():
    """
    Lädt den aktuellen Kontostand (virtuelles Geld zum Handeln).
    
    Rückgabe: Der Kontostand als Float (Dezimalzahl)
    Standardwert: 10000.0 USD (wenn keine Datei existiert)
    
    Der Nutzer startet also mit 10.000$ virtuellem Geld.
    """
    if BALANCE_FILE.exists():
        try:
            return float(json.loads(BALANCE_FILE.read_text(encoding="utf-8")))
        except:
            # Bei Fehlern: Standardwert zurückgeben
            return 10000.0
    # Wenn keine Datei existiert: Startwert 10.000$
    return 10000.0


def save_balance(balance):
    """
    Speichert den aktuellen Kontostand in die JSON-Datei.
    
    Parameter:
    - balance: Der neue Kontostand als Zahl (int oder float)
    """
    BALANCE_FILE.write_text(json.dumps(balance), encoding="utf-8")

# ================================================================================
# HILFSFUNKTIONEN: AKTIENDATEN VON YAHOO FINANCE ABRUFEN
# ================================================================================
# Diese Funktionen holen Echtzeit-Daten von Yahoo Finance.
# yfinance ist eine kostenlose Bibliothek für den Zugriff auf Finanzdaten.
# ================================================================================

def fetch_price(symbol):
    """
    Ruft den aktuellen Kurs und den Schlusskurs des Vortags für eine Aktie ab.
    
    Parameter:
    - symbol: Das Börsensymbol der Aktie (z.B. "AAPL" für Apple, "TSLA" für Tesla)
    
    Rückgabe: Tuple (aktueller_preis, vorheriger_schlusskurs)
              Beide Werte können None sein, wenn keine Daten verfügbar sind.
    
    Beispiel: fetch_price("AAPL") könnte (175.50, 174.20) zurückgeben
    """
    try:
        # Erstelle ein Ticker-Objekt für das Symbol
        # Ein Ticker ist wie ein "Handle" für alle Daten zu einer Aktie
        t = yf.Ticker(symbol)
        
        # fast_info enthält schnell abrufbare Basisdaten
        fast = getattr(t, "fast_info", None)
        
        if fast:
            # Hole letzten Preis und vorherigen Schlusskurs
            price = getattr(fast, "last_price", None)
            prev = getattr(fast, "previous_close", None)
            return price, prev
    except:
        # Bei Netzwerkfehlern oder ungültigen Symbolen: None zurückgeben
        pass
    return None, None


def fetch_name(symbol):
    """
    Ruft den vollständigen Firmennamen für ein Aktien-Symbol ab.
    
    Parameter:
    - symbol: Das Börsensymbol (z.B. "AAPL")
    
    Rückgabe: Der Firmenname (z.B. "Apple Inc.") oder das Symbol selbst als Fallback
    """
    try:
        t = yf.Ticker(symbol)
        # info enthält detaillierte Informationen zur Aktie
        info = t.info
        # Versuche zuerst longName, dann shortName, sonst das Symbol selbst
        return info.get("longName") or info.get("shortName") or symbol
    except:
        return symbol


def fetch_stock_history(symbol, period="1mo", interval="1d"):
    """
    Ruft historische Kursdaten für eine Aktie ab.
    
    Parameter:
    - symbol: Das Börsensymbol
    - period: Zeitraum der Daten. Mögliche Werte:
              "1d" (1 Tag), "5d" (5 Tage), "1mo" (1 Monat), "3mo" (3 Monate),
              "6mo" (6 Monate), "1y" (1 Jahr), "5y" (5 Jahre), "max" (alle Daten)
    - interval: Zeitabstand zwischen Datenpunkten:
                "1m" (1 Minute), "5m" (5 Min), "15m", "1h", "1d" (1 Tag), "1wk" (1 Woche)
    
    Rückgabe: Ein Pandas DataFrame mit Spalten:
              Open, High, Low, Close, Volume (Eröffnung, Hoch, Tief, Schluss, Volumen)
              Der Index ist das Datum/die Zeit.
    """
    try:
        t = yf.Ticker(symbol)
        # history() ruft die historischen Daten ab
        hist = t.history(period=period, interval=interval)
        return hist
    except:
        return None

def search_stocks(query):
    """
    Sucht nach Aktien basierend auf einem Suchbegriff (Name oder Symbol).
    
    Diese Funktion verwendet die Yahoo Finance Such-API, um passende
    Aktien, ETFs, Indizes oder Kryptowährungen zu finden.
    
    Parameter:
    - query: Der Suchbegriff (z.B. "Apple", "Tesla", "AAPL")
    
    Rückgabe: Eine Liste von Dictionaries mit gefundenen Wertpapieren:
    [{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"}, ...]
    
    Wenn die Suche fehlschlägt oder nichts gefunden wird: leere Liste []
    """
    # Mindestens 2 Zeichen benötigt für sinnvolle Suche
    if not query or len(query) < 2:
        return []
    
    try:
        # Yahoo Finance Such-API URL
        # quotesCount=10: Maximal 10 Ergebnisse
        # newsCount=0: Keine News-Ergebnisse (nur Wertpapiere)
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        
        # HTTP GET-Anfrage mit Timeout von 5 Sekunden
        # User-Agent Header simuliert einen normalen Browser
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        
        # JSON-Antwort parsen
        data = resp.json()
        
        results = []
        # Durchlaufe alle gefundenen "quotes" (Wertpapiere)
        for q in data.get("quotes", []):
            # Filtere nach Wertpapiertypen, die wir unterstützen
            # EQUITY = Aktie, ETF = Börsengehandelter Fonds, etc.
            if q.get("quoteType") in ["EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "CURRENCY"]:
                results.append({
                    "symbol": q.get("symbol"),
                    "name": q.get("shortname") or q.get("longname") or q.get("symbol"),
                    "exchange": q.get("exchange", "")  # Börse (z.B. NASDAQ, NYSE)
                })
        return results
    except:
        # Bei Netzwerkfehlern: leere Liste zurückgeben
        return []

def fetch_google_news(symbol, limit=20):
    """
    Ruft aktuelle Nachrichten zu einer Aktie von Google News ab.
    
    Die Funktion nutzt den RSS-Feed von Google News, um deutschsprachige
    Nachrichten zu einem bestimmten Suchbegriff (meist Aktien-Symbol) zu holen.
    
    Parameter:
    - symbol: Der Suchbegriff (z.B. "AAPL", "Tesla", "Bitcoin")
    - limit: Maximale Anzahl der zurückgegebenen Nachrichten (Standard: 20)
    
    Rückgabe: Eine Liste von Dictionaries mit News-Daten:
    [{"title": "Schlagzeile...", "link": "https://...", 
      "pubDate": "Mon, 15 Jan 2024...", "source": "Handelsblatt",
      "symbol": "AAPL"}, ...]
    
    RSS (Really Simple Syndication) ist ein XML-Format für News-Feeds.
    """
    try:
        # Google News RSS-Feed URL
        # hl=de: Sprache Deutsch
        # gl=DE: Region Deutschland
        # ceid=DE:de: Ländercode
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=de&gl=DE&ceid=DE:de"
        
        # RSS-Feed abrufen
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        
        # Alle <item>-Tags finden (jeder <item> ist eine Nachricht)
        # re.findall sucht alle Vorkommen des Musters
        # re.DOTALL lässt '.' auch Zeilenumbrüche matchen
        items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
        
        news = []
        # Verarbeite die gefundenen Items (maximal 'limit' Stück)
        for item in items[:limit]:
            # Extrahiere Titel, Link, Veröffentlichungsdatum und Quelle
            # mit Regular Expressions (Muster-Suche)
            title_m = re.search(r"<title>(.*?)</title>", item)
            link_m = re.search(r"<link>(.*?)</link>", item)
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item)
            source_m = re.search(r"<source.*?>(.*?)</source>", item)
            
            # Wenn gefunden, extrahiere den Text (group(1)), sonst Fallback-Wert
            # unescape() wandelt HTML-Entities zurück (z.B. &amp; -> &)
            title = unescape(title_m.group(1)) if title_m else "News"
            link = link_m.group(1) if link_m else ""
            pub = pub_m.group(1) if pub_m else ""
            source = unescape(source_m.group(1)) if source_m else ""
            
            news.append({
                "title": title,
                "link": link,
                "pubDate": pub,
                "source": source,
                "symbol": symbol  # Füge Symbol hinzu, damit wir wissen, zu welcher Aktie die News gehört
            })
        return news
    except:
        # Bei Fehlern: leere Liste zurückgeben
        return []

def format_volume(vol):
    """
    Formatiert große Zahlen (Handelsvolumen) in lesbare Kurzform.
    
    Handelsvolumen sind oft sehr große Zahlen (Millionen oder Milliarden).
    Diese Funktion wandelt sie in lesbare Kürzel um.
    
    Parameter:
    - vol: Das Handelsvolumen als Zahl (oder None)
    
    Rückgabe: Formatierte Zeichenkette
    
    Beispiele:
    - 1234567890 -> "1.23B" (Milliarden/Billions)
    - 5678000 -> "5.68M" (Millionen)
    - 45000 -> "45.0K" (Tausend)
    - 500 -> "500"
    - None -> "n/a"
    """
    if vol is None:
        return "n/a"  # "not available" - nicht verfügbar
    
    # Prüfe in absteigender Größenordnung
    if vol >= 1_000_000_000:  # >= 1 Milliarde (1B)
        return f"{vol/1_000_000_000:.2f}B"
    if vol >= 1_000_000:       # >= 1 Million (1M)
        return f"{vol/1_000_000:.2f}M"
    if vol >= 1_000:           # >= 1 Tausend (1K)
        return f"{vol/1_000:.1f}K"
    
    # Kleine Zahlen einfach als String zurückgeben
    return str(vol)

# ================================================================================
# HILFSFUNKTIONEN: DIAGRAMME/CHARTS ERSTELLEN
# ================================================================================
# Diese Funktionen erstellen interaktive Plotly-Diagramme für die Visualisierung.
# Plotly ist eine Bibliothek für interaktive, webfähige Graphen.
# ================================================================================

def create_stock_chart(symbol, period="1mo", interval="1d"):
    """
    Erstellt ein Liniendiagramm (Kursverlauf) für eine Aktie.
    
    Parameter:
    - symbol: Das Aktien-Symbol (z.B. "AAPL")
    - period: Zeitraum (z.B. "1d", "1mo", "1y")
    - interval: Datenintervall (z.B. "5m", "1d")
    
    Rückgabe: Ein Plotly Figure-Objekt (das fertige Diagramm)
    
    Features:
    - Grüne Linie wenn Kurs gestiegen, rote wenn gefallen
    - Leichte Färbung unter der Kurslinie
    - Prozentuale Veränderung im Titel
    - Optimale Y-Achsen-Skalierung
    """
    # Hole historische Kursdaten
    hist = fetch_stock_history(symbol, period, interval)
    
    # Wenn keine Daten vorhanden: zeige Fehlermeldung im Chart
    if hist is None or hist.empty:
        fig = go.Figure()
        # Füge eine Text-Annotation in der Mitte des Charts hinzu
        fig.add_annotation(text="Keine Daten verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        # Verstecke die Achsen bei leerem Chart
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    
    # Berechne ob der Kurs gestiegen oder gefallen ist
    start_price = hist["Close"].iloc[0]   # Erster Schlusskurs (iloc[0] = erste Zeile)
    end_price = hist["Close"].iloc[-1]    # Letzter Schlusskurs (iloc[-1] = letzte Zeile)
    is_positive = end_price >= start_price  # True wenn Kurs gestiegen
    
    # Wähle Farbe basierend auf Kursentwicklung
    # #22c55e = Grün (positiv), #ef4444 = Rot (negativ)
    color = "#22c55e" if is_positive else "#ef4444"
    
    # ===== Y-Achsen-Skalierung berechnen =====
    # Wir wollen die Y-Achse optimal zoomen, damit der Kursverlauf gut sichtbar ist
    y_min = hist["Close"].min()  # Tiefster Kurs
    y_max = hist["Close"].max()  # Höchster Kurs
    y_range = y_max - y_min      # Spannweite
    
    # Berechne Padding (Abstand am Rand) für bessere Optik
    if y_range < 0.01 * y_max:   # Wenn Kurs sehr stabil (wenig Schwankung)
        padding = 0.005 * y_max  # Kleines Padding
    else:
        padding = y_range * 0.1  # 10% der Spannweite als Padding
    
    # ===== Diagramm erstellen =====
    fig = go.Figure()
    
    # Füge die Kurslinie hinzu
    fig.add_trace(go.Scatter(
        x=hist.index,              # X-Achse: Datum/Zeit
        y=hist["Close"],           # Y-Achse: Schlusskurse
        mode="lines",              # Nur Linien, keine Punkte
        line=dict(color=color, width=2),  # Linienfarbe und -dicke
        fill="tozeroy",            # Fülle den Bereich bis zur X-Achse
        # Halbtransparente Füllfarbe (RGBA: Rot, Grün, Blau, Alpha/Transparenz)
        fillcolor=f"rgba({34 if is_positive else 239}, {197 if is_positive else 68}, {94 if is_positive else 68}, 0.1)",
        name=symbol,               # Name für die Legende
        hovertemplate="%{y:.2f}<extra></extra>"  # Tooltip-Format (2 Nachkommastellen)
    ))
    
    # Berechne prozentuale Veränderung für den Titel
    pct_change = ((end_price - start_price) / start_price) * 100
    sign = "+" if pct_change >= 0 else ""  # Pluszeichen bei positiven Werten
    
    # ===== Layout/Design des Diagramms anpassen =====
    fig.update_layout(
        # Titel mit Symbol und prozentualer Veränderung
        title=dict(text=f"{symbol} ({sign}{pct_change:.2f}%)", font=dict(size=16, color=color)),
        # Y-Achse: Bereich, Zahlenformat, Gitternetzfarbe
        yaxis=dict(range=[y_min - padding, y_max + padding], tickformat=",.2f", gridcolor="#e5e7eb"),
        # X-Achse: Gitternetzlinien anzeigen
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        plot_bgcolor="white",       # Hintergrundfarbe des Plots
        paper_bgcolor="white",      # Hintergrundfarbe außerhalb des Plots
        margin=dict(l=50, r=20, t=50, b=50),  # Ränder (left, right, top, bottom)
        hovermode="x unified",      # Tooltip-Modus: einheitlich für X-Position
        showlegend=False            # Keine Legende anzeigen
    )
    return fig

def create_portfolio_pie_chart(portfolio):
    """
    Erstellt ein Kreisdiagramm (Pie Chart) der Portfolio-Zusammensetzung.
    
    Das Diagramm zeigt, wie das investierte Geld auf verschiedene
    Aktien verteilt ist (in Prozent und absoluten Werten).
    
    Parameter:
    - portfolio: Liste der Portfolio-Positionen aus der JSON-Datei
    
    Rückgabe: Ein Plotly Figure-Objekt (Kreisdiagramm)
    """
    # Wenn Portfolio leer ist, zeige Nachricht
    if not portfolio:
        fig = go.Figure()
        fig.add_annotation(text="Portfolio ist leer", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    try:
        labels = []   # Namen/Symbole für das Diagramm
        values = []   # Werte (aktueller Wert der Position)
        colors = []   # Farben für jeden Sektor
        
        # Durchlaufe alle Positionen im Portfolio
        for item in portfolio:
            symbol = item["symbol"]
            qty = item["qty"]  # Anzahl der Aktien
            
            # Hole aktuellen Preis
            current_price, _ = fetch_price(symbol)
            
            if current_price:
                # Berechne aktuellen Wert der Position
                value = qty * current_price
                values.append(value)
                labels.append(symbol)
                
                # Generiere eine Farbe basierend auf dem Symbol
                # hash() erzeugt eine Zahl aus dem Symbol, % 360 gibt uns einen Farbwinkel (HSL)
                colors.append(f"hsl({hash(symbol) % 360}, 70%, 50%)")
        
        # Wenn keine Preise verfügbar
        if not values:
            fig = go.Figure()
            fig.add_annotation(text="Keine aktuellen Preise verfügbar", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        
        # Erstelle das Kreisdiagramm
        fig = go.Figure(data=[go.Pie(
            labels=labels,              # Beschriftungen der Sektoren
            values=values,              # Werte (bestimmen Größe der Sektoren)
            marker_colors=colors,       # Farben der Sektoren
            textinfo='label+percent',   # Zeige Label und Prozentwert
            insidetextorientation='radial',  # Text radial ausrichten
            textfont=dict(color="white")     # Weiße Schrift
        )])
        
        # Layout anpassen
        fig.update_layout(
            title=dict(text="Portfolio-Zusammensetzung", font=dict(color="white")),
            showlegend=False,  # Keine separate Legende (Labels sind im Chart)
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",  # Transparenter Hintergrund
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    except Exception:
        # Bei Fehlern: Fehler-Chart anzeigen
        fig = go.Figure()
        fig.add_annotation(text="Fehler beim Laden", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

def create_portfolio_value_chart(portfolio):
    """
    Erstellt ein Liniendiagramm zur Visualisierung der Portfolio-Wertentwicklung.
    
    Dieses Diagramm zeigt für jede Aktie im Portfolio:
    - Den Kaufwert (investiertes Geld) als gestrichelte Linie
    - Den aktuellen Wert als durchgezogene Linie
    - Grüne Bereiche bei Gewinn, rote bei Verlust
    - Gewinn/Verlust-Annotationen über jeder Position
    
    Parameter:
    - portfolio: Liste der Portfolio-Positionen
    
    Rückgabe: Ein Plotly Figure-Objekt
    """
    
    # Hilfsfunktion für leere/Fehler-Charts
    def empty_fig(text):
        """Erstellt ein leeres Diagramm mit einer Nachricht."""
        fig = go.Figure()
        fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    # Prüfe ob Portfolio Daten enthält
    if not portfolio:
        return empty_fig("Portfolio ist leer")
    
    try:
        # ===== Daten sammeln =====
        data_list = []      # Liste für alle Portfolio-Daten
        total_invested = 0  # Summe aller investierten Beträge
        total_current = 0   # Summe aller aktuellen Werte
        
        # Durchlaufe jede Position im Portfolio
        for item in portfolio:
            symbol = item.get("symbol", "")
            qty = item.get("qty", 0)  # Anzahl der Aktien
            
            # Kaufpreis: Versuche "buy_price", sonst "avg_price"
            buy_price = item.get("buy_price") or item.get("avg_price", 0)
            
            # Berechne investierten Betrag
            invested = qty * buy_price
            total_invested += invested
            
            # Hole aktuellen Preis
            try:
                current_price, _ = fetch_price(symbol)
            except:
                current_price = None
            
            # Wenn aktueller Preis verfügbar, berechne Statistiken
            if current_price:
                current_value = qty * current_price  # Aktueller Wert
                total_current += current_value
                pnl = current_value - invested       # Profit and Loss (Gewinn/Verlust)
                
                # Prozentuale Veränderung berechnen (Vorsicht: Division durch 0 vermeiden)
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                
                # Hole Firmennamen
                try:
                    name = fetch_name(symbol) or symbol
                except:
                    name = symbol
                
                # Füge alle Daten zur Liste hinzu
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
        
        # ===== Daten für das Diagramm vorbereiten =====
        # Kürze lange Namen auf 15 Zeichen
        symbols = [d["name"][:15] if d["name"] else d["symbol"] for d in data_list]
        invested_values = [d["invested"] for d in data_list]
        current_values = [d["current"] for d in data_list]
        pnl_values = [d["pnl"] for d in data_list]
        
        # ===== Diagramm erstellen =====
        fig = go.Figure()
        
        # ===== Füllbereiche für Gewinn/Verlust erstellen =====
        # Für jede Position einen farbigen Bereich zwischen Kaufwert und aktuellem Wert
        for i, d in enumerate(data_list):
            # Erstelle ein Rechteck um die Position (0.3 Einheiten breit)
            x_pos = [i - 0.3, i + 0.3, i + 0.3, i - 0.3]
            
            if d["current"] >= d["invested"]:
                # ===== GRÜNER BEREICH (Gewinn) =====
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
                # ===== ROTER BEREICH (Verlust) =====
                y_fill = [d["current"], d["current"], d["invested"], d["invested"]]
                fig.add_trace(go.Scatter(
                    x=x_pos, y=y_fill,
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.3)",  # Halbtransparentes Rot
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
        # ===== Kaufwert-Linie (gestrichelt, gelb) =====
        fig.add_trace(go.Scatter(
            x=list(range(len(symbols))),  # X-Positionen: 0, 1, 2, ...
            y=invested_values,
            mode="lines+markers",          # Linie mit Punkten
            name="Kaufwert",
            line=dict(color="#fbbf24", width=3, dash="dash"),  # Gelb, gestrichelt
            marker=dict(size=10, symbol="diamond"),  # Diamant-Marker
            hovertemplate="<b>Kaufwert</b><br>%{y:,.2f} USD<extra></extra>"
        ))
        
        # ===== Aktueller Wert-Linie (durchgezogen, blau mit farbigen Punkten) =====
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
        
        # ===== Gewinn/Verlust-Annotationen über den Positionen =====
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
        
        # ===== Gesamt P/L berechnen =====
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        total_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        total_sign = "+" if total_pnl >= 0 else ""
        
        # ===== Layout konfigurieren =====
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


# ================================================================================
# APP INITIALISIERUNG
# ================================================================================
# Hier wird die Dash-Anwendung erstellt und konfiguriert.
# Dash ist das Framework, das unsere Web-App antreibt.
# ================================================================================

# Erstelle die Dash-App
app = dash.Dash(
    __name__,  # Name des Moduls (wird für Pfade verwendet)
    external_stylesheets=[
        dbc.themes.DARKLY,  # Bootstrap Dark Theme für ein modernes dunkles Design
        # FontAwesome für Icons (wie ❤, ⭐, etc. aber als Vektorgrafiken)
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ], 
    suppress_callback_exceptions=True  # Unterdrückt Fehler bei dynamischen Callbacks
)

# Titel der Webseite (erscheint im Browser-Tab)
app.title = "Stock Dashboard"

# ================================================================================
# CUSTOM CSS (Benutzerdefinierte Styles)
# ================================================================================
# app.index_string ermöglicht es uns, das gesamte HTML-Template der Seite anzupassen.
# Hier fügen wir eigenes CSS hinzu für:
# - News-Kacheln mit Hover-Effekten
# - Light Mode / Dark Mode Unterstützung
# - Benutzerdefinierte Tabellenformatierung
# - Theme-Toggle-Button-Animation
#
# CSS (Cascading Style Sheets) bestimmt das Aussehen der HTML-Elemente.
# Die Syntax ist: selector { eigenschaft: wert; }
# ================================================================================

app.index_string = '''
<!DOCTYPE html>
<html>
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

# ================================================================================
# LAYOUT - Die visuelle Struktur der Anwendung
# ================================================================================
# Das Layout definiert, was der Benutzer sieht und womit er interagieren kann.
# Es besteht aus verschachtelten HTML- und Dash-Komponenten.
#
# Wichtige Konzepte:
# - dbc.Container: Ein Bootstrap-Container, der die Seite strukturiert
# - dbc.Row/dbc.Col: Zeilen und Spalten für das Grid-Layout
# - html.Div: Ein HTML-Div-Element (Container für andere Elemente)
# - dbc.Button: Ein klickbarer Button
# - dcc.Input: Ein Eingabefeld
# - dcc.Graph: Ein Plotly-Diagramm
# - dbc.Modal: Ein Pop-up Fenster
# - dcc.Store: Unsichtbarer Datenspeicher (für den Client-seitigen Zustand)
# ================================================================================

def create_market_ticker():
    """
    Erstellt die Marktübersicht-Leiste (Ticker-Bar) oben auf der Seite.
    
    Diese Leiste zeigt die wichtigsten Marktindizes und deren aktuelle Kurse an.
    Jedes Element ist klickbar und öffnet ein Detail-Modal.
    
    Rückgabe: Eine dbc.Row mit den Ticker-Elementen
    """
    return dbc.Row([
        # Für jedes Symbol in der MARKET_OVERVIEW_SYMBOLS-Liste ein Element erstellen
        # List Comprehension: Erstellt eine Liste von dbc.Col-Elementen
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


# ================================================================================
# HAUPT-LAYOUT DER ANWENDUNG
# ================================================================================
# app.layout definiert die gesamte Struktur der Web-Anwendung.
# Hier werden alle UI-Elemente definiert und angeordnet.
# ================================================================================

app.layout = dbc.Container([
    # ===== INTERVALL-TIMER FÜR AUTOMATISCHE UPDATES =====
    # Dieser Intervall-Timer löst alle 15 Sekunden ein Update aus
    # n_intervals zählt hoch und kann als Input für Callbacks verwendet werden
    dcc.Interval(id="market-interval", interval=15000, n_intervals=0),  # 15000ms = 15 Sekunden
    
    # ===== DATENSPEICHER (STORES) =====
    # dcc.Store speichert Daten im Browser des Benutzers
    # Diese Daten überleben Seitenaktualisierungen und können von Callbacks gelesen werden
    dcc.Store(id="selected-ticker", data=None),           # Aktuell ausgewählte Aktie für Buy/Sell
    dcc.Store(id="portfolio-store", data=load_portfolio()),  # Portfolio-Daten (geladen aus Datei)
    dcc.Store(id="search-results-store", data=[]),        # Suchergebnisse
    dcc.Store(id="theme-store", data="dark"),             # Aktuelles Theme (dark/light)
    
    # ===== HEADER MIT TITEL UND THEME-TOGGLE =====
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
    
    # Unsichtbares Div für Theme-Toggle Output (wird von JavaScript gesteuert)
    html.Div(id="theme-output", style={"display": "none"}),
    
    # ===== MARKT-TICKER-LEISTE =====
    create_market_ticker(),
    
    # ===== HAUPT-TABS =====
    # Tabs sind wie Karteireiter - der Benutzer kann zwischen verschiedenen Ansichten wechseln
    dbc.Tabs([
        # ================================================================
        # TAB 1: PORTFOLIO - Zeigt alle gekauften Aktien und deren Wert
        # ================================================================
        dbc.Tab(label="Portfolio", children=[
            # Sub-Tabs innerhalb des Portfolio-Tabs
            dbc.Tabs([
                # ----- ÜBERSICHT SUB-TAB -----
                # Zeigt Tabelle mit allen Positionen, Zusammenfassung und Kreisdiagramm
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
                    # Platzhalter für die Portfolio-Tabelle (wird durch Callback gefüllt)
                    html.Div(id="portfolio-table"),
                    # Platzhalter für die Zusammenfassung (Investiert, Wert, P/L)
                    html.Div(id="portfolio-summary", className="mt-3"),
                    # Kreisdiagramm der Portfolio-Zusammensetzung
                    dbc.Row([
                        dbc.Col([dcc.Graph(id="portfolio-chart", style={"height": "400px"})], width=12),
                    ], className="mt-3"),
                ], className="p-3"),  # p-3 = Padding
                
                # ----- WERTENTWICKLUNG SUB-TAB -----
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
        
        # ================================================================
        # TAB 2: AKTIEN - Aktiensuche, Kurs-Charts und News
        # ================================================================
        # Hier kann der Benutzer nach Aktien suchen, deren Kursverlauf
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
        
        # ================================================================
        # TAB 3: NEWS - Aktuelle Finanznachrichten
        # ================================================================
        # Zeigt Nachrichten aus verschiedenen Kategorien (Aktien, Krypto, Wirtschaft)
        # in einem schönen Kachel-Layout mit Bildern.
        dbc.Tab(label="📰 News", children=[
            html.Div([
                # ----- HEADER MIT SUCHE UND AKTUALISIEREN-BUTTON -----
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
                
                # ----- KATEGORIE-TABS -----
                # Unter-Tabs zum Filtern nach Kategorie
                dbc.Tabs([
                    dbc.Tab(label="🌍 Alle", tab_id="news-all"),       # Alle Nachrichten
                    dbc.Tab(label="📈 Aktien", tab_id="news-stocks"),  # Nur Aktien-News
                    dbc.Tab(label="₿ Krypto", tab_id="news-crypto"),    # Nur Krypto-News
                    dbc.Tab(label="🏦 Wirtschaft", tab_id="news-economy"),  # Wirtschaftsnews
                ], id="news-category-tabs", active_tab="news-all", className="mb-4"),
                
                # ----- NEWS-KACHELN CONTAINER -----
                # dbc.Spinner zeigt einen Ladekreis während die News geladen werden
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
    
], fluid=True)  # fluid=True = Container nimmt volle Breite ein


# ================================================================================
# CALLBACKS - Die Interaktionslogik der Anwendung
# ================================================================================
# Callbacks sind das Herzstück von Dash-Anwendungen. Sie verbinden Benutzeraktionen
# (wie Klicks oder Texteingaben) mit Reaktionen der App (wie Aktualisieren von
# Diagrammen oder Texten).
#
# WICHTIGE KONZEPTE:
# ------------------
# 1. @callback Dekorator: Markiert eine Funktion als Callback
#
# 2. Output("element-id", "eigenschaft"): Was soll aktualisiert werden?
#    - "element-id": Die ID des HTML-Elements (z.B. "stock-chart")
#    - "eigenschaft": Was am Element geändert wird (z.B. "figure", "children")
#
# 3. Input("element-id", "eigenschaft"): Was löst den Callback aus?
#    - Wenn sich diese Eigenschaft ändert, wird der Callback ausgeführt
#
# 4. State("element-id", "eigenschaft"): Zusätzliche Daten lesen
#    - Wie Input, aber löst den Callback NICHT aus
#    - Nützlich um zusätzliche Werte zu lesen
#
# 5. prevent_initial_call=True: Callback wird nicht beim Laden der Seite ausgeführt
#
# 6. ctx.triggered_id: Welches Element hat den Callback ausgelöst?
#
# Beispiel:
# @callback(
#     Output("mein-text", "children"),    # Aktualisiert den Inhalt von "mein-text"
#     Input("mein-button", "n_clicks")    # Wird ausgeführt wenn Button geklickt wird
# )
# def meine_funktion(n_clicks):
#     return f"Button wurde {n_clicks} mal geklickt"
# ================================================================================


# ================================================================================
# CALLBACK: MARKT-TICKER AKTUALISIEREN
# ================================================================================
# Dieser Callback aktualisiert die Kurse in der oberen Ticker-Leiste.
# Er wird alle 15 Sekunden automatisch durch den Interval-Timer ausgelöst.
@callback(
    # OUTPUTS: Aktualisiere Text UND Style für jedes Ticker-Element
    # List Comprehension erstellt eine Liste von Outputs für alle Symbole
    [Output(f"ticker-{s['name']}", "children") for s in MARKET_OVERVIEW_SYMBOLS] +
    [Output(f"ticker-{s['name']}", "style") for s in MARKET_OVERVIEW_SYMBOLS],
    
    # INPUT: Der Interval-Timer (aktualisiert sich alle 15 Sekunden)
    Input("market-interval", "n_intervals")
)
def update_market_tickers(n):
    """
    Aktualisiert alle Kurse in der Marktübersicht-Leiste.
    
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
        
        # Spezialfall EUR/USD: Invertieren (weil Yahoo USD/EUR liefert)
        if s.get("invert") and price:
            price = 1 / price
            if prev:
                prev = 1 / prev
        
        # Wenn kein Preis verfügbar: "n/a" anzeigen
        if price is None:
            texts.append(html.Span([html.B(s["name"]), ": n/a"]))
            styles.append({"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa", "padding": "8px"})
        else:
            # Formatiere den Preis mit der konfigurierten Anzahl Nachkommastellen
            decimals = s.get("decimals", 2)
            # Deutsche Zahlenformatierung: 1.234,56 statt 1,234.56
            formatted = f"{price:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            # Farbe basierend auf Änderung zum Vortag
            if prev:
                diff = price - prev
                # Grün wenn gestiegen, Rot wenn gefallen, Schwarz wenn gleich
                color = "#22c55e" if diff > 0.0001 else "#ef4444" if diff < -0.0001 else "#000000"
            else:
                color = "#000000"
            
            texts.append(html.Span([html.B(s["name"]), f": {formatted}"], style={"color": color, "fontWeight": "bold"}))
            styles.append({"cursor": "pointer", "borderRadius": "5px", "background": "#f8f9fa", "padding": "8px"})
    
    # Rückgabe: Erst alle Texte, dann alle Styles (entspricht der Output-Reihenfolge)
    return texts + styles

# ================================================================================
# CALLBACK: AKTIENSUCHE UND CHART-AKTUALISIERUNG
# ================================================================================
# Dieser Callback reagiert auf:
# 1. Texteingabe im Suchfeld
# 2. Klicks auf die Zeitraum-Buttons (1T, 1W, 1M, etc.)
@callback(
    Output("stock-chart", "figure"),   # Aktualisiert das Diagramm
    Output("stock-news", "children"),   # Aktualisiert die News-Liste
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
    """
    Aktualisiert den Aktienchart und die News basierend auf der Suche
    und dem gewählten Zeitraum.
    
    Parameter:
    - search: Der Suchbegriff aus dem Eingabefeld
    - n1d bis nmax: Klick-Zähler der Zeitraum-Buttons (Werte werden nicht direkt
                    verwendet, aber der Klick löst den Callback aus)
    
    Rückgabe: (chart_figure, news_elements)
    """
    # Finde heraus, welches Element den Callback ausgelöst hat
    triggered = ctx.triggered_id
    
    # Mapping: Button-ID -> (Periode, Interval)
    # Periode = wie weit zurück, Interval = Abstand zwischen Datenpunkten
    period_map = {
        "btn-1d": ("1d", "5m"),      # 1 Tag, alle 5 Minuten
        "btn-1w": ("5d", "15m"),     # 5 Tage, alle 15 Minuten
        "btn-1m": ("1mo", "1d"),     # 1 Monat, täglich
        "btn-3m": ("3mo", "1d"),     # 3 Monate, täglich
        "btn-1y": ("1y", "1wk"),     # 1 Jahr, wöchentlich
        "btn-max": ("max", "1mo"),   # Maximum, monatlich
    }
    
    # Hole Periode und Interval basierend auf geklicktem Button
    # Falls keiner geklickt: Standard ist 1 Monat
    period, interval = period_map.get(triggered, ("1mo", "1d"))
    
    # Prüfe ob Suchbegriff lang genug ist
    if not search or len(search) < 2:
        return go.Figure(), html.P("Bitte Aktie suchen...")
    
    # Suche nach passenden Aktien
    results = search_stocks(search)
    if not results:
        return go.Figure(), html.P("Keine Ergebnisse")
    
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
    
    return fig, news_items

# ================================================================================
# CALLBACK: PORTFOLIO ANZEIGE AKTUALISIEREN
# ================================================================================
# Dieser Callback aktualisiert die gesamte Portfolio-Übersicht wenn sich die
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
    """
    Aktualisiert alle Portfolio-Ansichten: Tabelle, Zusammenfassung, Charts.
    
    Parameter:
    - portfolio: Liste der Portfolio-Positionen aus dem Store
    
    Rückgabe: (tabelle, zusammenfassung, kreisdiagramm, aktienkacheln, gesamtsummary)
    """
    
    # Hilfsfunktion für leere/Fehler-Charts
    def empty_figure(text="Portfolio ist leer"):
        """Erstellt ein leeres Diagramm mit einer Nachricht."""
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


# ================================================================================
# CALLBACK: BUY/SELL MODAL ÖFFNEN/SCHLIESSEN
# ================================================================================
# Modals sind Pop-up-Fenster, die über der Seite erscheinen.
# Dieser Callback steuert, ob das Kauf/Verkauf-Modal sichtbar ist.
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
    """
    Wechselt den Zustand des Buy/Sell-Modals (offen <-> geschlossen).
    
    Parameter:
    - n1-n4: Klick-Zähler der verschiedenen Buttons
    - is_open: Aktueller Zustand (True = offen, False = geschlossen)
    
    Rückgabe: Der neue Zustand (invertiert)
    """
    return not is_open  # Einfach umschalten: offen -> zu, zu -> offen


# ================================================================================
# CALLBACK: KONTOSTAND MODAL ÖFFNEN/SCHLIESSEN
# ================================================================================
@callback(
    Output("kontostand-modal", "is_open"),
    Input("btn-kontostand", "n_clicks"),     # "Kontostand" Button
    Input("btn-close-kontostand", "n_clicks"), # "Schließen" Button
    State("kontostand-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_kontostand_modal(n1, n2, is_open):
    """Wechselt den Zustand des Kontostand-Modals."""
    return not is_open


# ================================================================================
# CALLBACK: KONTOSTAND ANZEIGEN UND EIN-/AUSZAHLUNG VERARBEITEN
# ================================================================================
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
    Verarbeitet Kontostand-Anzeige und Ein-/Auszahlungen.
    
    Parameter:
    - is_open: Ob das Modal gerade geöffnet wurde
    - btn_deposit: Klick-Zähler Einzahlung-Button
    - btn_withdraw: Klick-Zähler Auszahlung-Button
    - amount: Der eingegebene Betrag
    
    Rückgabe: (kontostand_anzeige, nachricht)
    """
    # Finde heraus, welches Element den Callback ausgelöst hat
    triggered = ctx.triggered_id
    
    # Lade aktuellen Kontostand aus Datei
    balance = load_balance()
    
    # ===== Fall 1: Modal wurde geöffnet =====
    if triggered == "kontostand-modal":
        if is_open:
            # Zeige aktuellen Kontostand
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), ""
        return "", ""
    
    # ===== Fall 2: Einzahlung =====
    if triggered == "btn-deposit":
        # Validierung: Betrag muss positiv sein
        if not amount or amount <= 0:
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert("Ungültiger Betrag", color="danger")
        
        # Betrag zum Kontostand addieren
        balance += float(amount)
        save_balance(balance)
        return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert(f"Einzahlung von {amount:,.2f} USD erfolgreich!", color="success")
    
    # ===== Fall 3: Auszahlung =====
    if triggered == "btn-withdraw":
        # Validierung: Betrag muss positiv sein
        if not amount or amount <= 0:
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert("Ungültiger Betrag", color="danger")
        
        # Validierung: Genügend Guthaben vorhanden?
        if balance < float(amount):
            return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert("Nicht genügend Guthaben", color="danger")
        
        # Betrag vom Kontostand abziehen
        balance -= float(amount)
        save_balance(balance)
        return dbc.Alert(f"Aktueller Kontostand: {balance:,.2f} USD", color="success"), dbc.Alert(f"Auszahlung von {amount:,.2f} USD erfolgreich!", color="success")
    
    # Wenn keiner der Fälle zutrifft: Nichts tun
    raise dash.exceptions.PreventUpdate

# ================================================================================
# CALLBACK: AKTIENSUCHE IM BUY/SELL MODAL
# ================================================================================
# Sucht nach Aktien basierend auf der Eingabe im Modal und zeigt Ergebnisse.
@callback(
    Output("buy-search-results", "children"),  # Liste der Suchergebnis-Buttons
    Output("search-results-store", "data"),    # Speichert Ergebnisse für spätere Verwendung
    Input("buy-search", "value"),              # Suchfeld im Modal
    prevent_initial_call=True
)
def search_for_buy(query):
    """
    Sucht nach Aktien für den Kauf und erstellt klickbare Buttons.
    
    Parameter:
    - query: Der Suchbegriff aus dem Eingabefeld
    
    Rückgabe: (buttons_liste, suchergebnisse_daten)
    """
    # Mindestens 2 Zeichen für Suche erforderlich
    if not query or len(query) < 2:
        return [], []
    
    # Suche durchführen
    results = search_stocks(query)
    
    # Erstelle Buttons für jedes Ergebnis (maximal 5)
    # Das Pattern {"type": "search-result", "index": i} ermöglicht Pattern Matching Callbacks
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


# ================================================================================
# CALLBACK: AKTIE FÜR KAUF/VERKAUF AUSWÄHLEN
# ================================================================================
# Wird ausgelöst wenn der Benutzer auf einen Suchergebnis-Button klickt.
# Verwendet Pattern Matching: Input({"type": "search-result", "index": dash.ALL}, ...)
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
    Zeigt Details zur ausgewählten Aktie an wenn ein Suchergebnis geklickt wird.
    
    Parameter:
    - clicks: Liste aller Klick-Zähler (einer pro Suchergebnis-Button)
    - results: Die gespeicherten Suchergebnisse
    
    Rückgabe: (aktien_info, chart, ticker_daten)
    """
    # Prüfe ob überhaupt geklickt wurde
    if not any(clicks) or not results:
        return "", "", None
    
    # Finde den Index des geklickten Buttons
    # next() findet das erste Element, das die Bedingung erfüllt
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


# ================================================================================
# CALLBACK: KAUF BESTÄTIGEN
# ================================================================================
# Dieser Callback wird ausgeführt wenn der "Kaufen" Button geklickt wird.
# Er fügt die Aktie zum Portfolio hinzu und zieht den Betrag vom Kontostand ab.
@callback(
    Output("portfolio-store", "data", allow_duplicate=True),  # allow_duplicate weil mehrere Callbacks diesen Output haben
    Input("btn-confirm-buy", "n_clicks"),     # Kaufen-Button wurde geklickt
    State("selected-ticker", "data"),         # Die ausgewählte Aktie
    State("buy-qty", "value"),                # Die eingegebene Menge
    State("portfolio-store", "data"),         # Aktuelles Portfolio
    prevent_initial_call=True
)
def confirm_buy(n, ticker, qty, portfolio):
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
    """
    # Validierung: Alle erforderlichen Daten vorhanden?
    if not n or not ticker or not qty or not ticker.get("price"):
        return portfolio or []

    # Sicherheitsprüfung: Genug Geld vorhanden?
    balance = load_balance()
    total_cost = int(qty) * float(ticker.get("price", 0))

    if total_cost > balance:
        # Nicht genug Geld: Kauf abbrechen, Portfolio unverändert zurückgeben
        return portfolio or []
    
    portfolio = portfolio or []  # Falls None, leere Liste verwenden
    
    # ===== Portfolio aktualisieren =====
    # Prüfen ob diese Aktie schon im Portfolio ist
    found = False
    for item in portfolio:
        if item["symbol"] == ticker["symbol"]:
            # Aktie bereits vorhanden: Durchschnittlichen Kaufpreis berechnen
            # Formel: (alter_preis * alte_menge + neuer_preis * neue_menge) / gesamtmenge
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
    
    # Wenn Aktie noch nicht im Portfolio: Neue Position hinzufügen
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
    
    # ===== Kontostand aktualisieren =====
    # Kaufpreis vom Kontostand abziehen
    try:
        current_balance = load_balance()
        cost = int(qty) * float(ticker.get("price", 0))
        current_balance -= cost
        save_balance(current_balance)
    except Exception:
        # Bei Fehlern: Nichts weiter tun (Portfolio wurde trotzdem aktualisiert)
        pass

    return portfolio

# ================================================================================
# CALLBACK: VERKAUF BESTÄTIGEN
# ================================================================================
# Dieser Callback wird ausgeführt wenn der "Verkaufen" Button geklickt wird.
# Er entfernt Aktien aus dem Portfolio und fügt den Erlös zum Kontostand hinzu.
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
    Führt den Verkauf einer Aktie durch.
    
    Schritte:
    1. Validierung der Eingaben
    2. Position im Portfolio finden und reduzieren
    3. Bei Menge 0: Position komplett entfernen
    4. Transaktion speichern
    5. Erlös zum Kontostand addieren
    
    Parameter:
    - n: Klick-Zähler
    - ticker: Ausgewählte Aktie
    - qty: Zu verkaufende Menge
    - portfolio: Aktuelles Portfolio
    
    Rückgabe: Das aktualisierte Portfolio
    """
    # Validierung
    if not n or not ticker or not qty:
        return portfolio or []
    
    symbol = ticker["symbol"]
    qty = int(qty)
    portfolio = portfolio or []
    
    # ===== Position im Portfolio finden und aktualisieren =====
    for item in portfolio:
        if item["symbol"] == symbol:
            # Prüfen ob genug Aktien zum Verkaufen vorhanden sind
            if item["qty"] >= qty:
                item["qty"] -= qty  # Menge reduzieren
                
                # Wenn alle Aktien verkauft: Position entfernen
                if item["qty"] == 0:
                    portfolio.remove(item)
                break
    
    # Portfolio speichern
    save_portfolio(portfolio)
    
    # Transaktion in Historie speichern
    save_transaction({
        "timestamp": datetime.now().isoformat(),
        "type": "sell",  # Verkauf
        "symbol": symbol,
        "qty": qty,
        "price": ticker.get("price", 0)
    })
    
    # ===== Kontostand aktualisieren =====
    # Verkaufserlös zum Kontostand addieren
    try:
        current_balance = load_balance()
        proceeds = qty * float(ticker.get("price", 0))  # Erlös = Menge * Preis
        current_balance += proceeds
        save_balance(current_balance)
    except Exception:
        pass

    return portfolio

# Transactions Modal
# ================================================================================
# TRANSAKTIONS-HISTORIE CALLBACK
# ================================================================================
# Dieser Callback verwaltet das Transaktions-Modal, das alle vergangenen
# Käufe und Verkäufe anzeigt.
#
# Funktionen:
# - Öffnen/Schließen des Modals
# - Filtern nach Jahr, Monat und Transaktionstyp (Kauf/Verkauf)
# - Anzeige einer interaktiven Tabelle mit Sortier- und Filterfunktionen
# - Berechnung von Zusammenfassungs-Statistiken (Summe Käufe, Verkäufe, Saldo)
#
# Die Tabelle verwendet dash_table.DataTable für erweiterte Funktionen:
# - Pagination (seitenweise Anzeige)
# - Native Sortierung und Filterung
# - Bedingte Formatierung (grün für Käufe, rot für Verkäufe)
# ================================================================================
@callback(
    # --- Outputs: Was der Callback aktualisiert ---
    Output("transactions-modal", "is_open"),         # Modal öffnen/schließen
    Output("transactions-table", "children"),        # Tabellen-Inhalt
    Output("transactions-summary", "children"),      # Zusammenfassungs-Karten
    Output("tx-year", "options"),                    # Jahr-Dropdown-Optionen
    
    # --- Inputs: Was den Callback auslöst ---
    Input("btn-transactions", "n_clicks"),           # "Transaktionen"-Button
    Input("btn-close-tx", "n_clicks"),               # Schließen-Button
    Input("tx-year", "value"),                       # Jahr-Filter
    Input("tx-month", "value"),                      # Monat-Filter
    Input("tx-type", "value"),                       # Typ-Filter (Kauf/Verkauf)
    
    # --- State: Zusätzliche Daten ohne Trigger ---
    State("transactions-modal", "is_open"),          # Aktueller Modal-Status
    prevent_initial_call=True
)
def toggle_transactions(n1, n2, year, month, tx_type, is_open):
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
    """
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

# ================================================================================
# TICKER DETAIL MODAL - Market Overview Detailansicht
# ================================================================================
# Wenn der Benutzer auf einen Ticker in der Market-Overview-Leiste klickt,
# öffnet sich dieses Modal mit detaillierten Informationen.
#
# Features:
# - Zeigt Detailchart für den ausgewählten Index/ETF
# - Zeitraum-Buttons für verschiedene Ansichten (1 Tag, 1 Woche, 1 Monat, 3 Monate)
# - Aktuelle Statistiken (Kurs, Hoch, Tief, Volumen)
#
# Dynamische Inputs:
# - Die Inputs werden dynamisch für alle MARKET_OVERVIEW_SYMBOLS generiert
# - Das ermöglicht Klick-Handler für jeden einzelnen Ticker
# - List Comprehension erstellt für jedes Symbol einen eigenen Input
#
# *args Pattern:
# - Weil die Anzahl der Inputs variabel ist, werden alle als *args übergeben
# - Der Code muss dann die einzelnen Argumente extrahieren
# ================================================================================
@callback(
    # --- Outputs ---
    Output("ticker-modal", "is_open"),               # Modal öffnen/schließen
    Output("ticker-modal-header", "children"),       # Titel im Modal-Header
    Output("ticker-modal-stats", "children"),        # Statistik-Anzeige
    Output("ticker-modal-chart", "figure"),          # Chart-Figur
    Output("current-ticker-symbol", "data"),         # Speichert aktuelles Symbol
    
    # --- Dynamische Inputs für alle Ticker + Kontroll-Buttons ---
    # Für jedes Symbol in MARKET_OVERVIEW_SYMBOLS wird ein Input erstellt
    [Input(f"ticker-{s['name']}", "n_clicks") for s in MARKET_OVERVIEW_SYMBOLS] +
    # Plus die Kontroll-Buttons für Schließen und Zeitraum-Wechsel
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
    Verwaltet das Ticker-Detail-Modal.
    
    Verwendet *args weil die Anzahl der Inputs dynamisch ist
    (abhängig von der Anzahl der Symbole in MARKET_OVERVIEW_SYMBOLS).
    """
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

# ================================================================================
# SENTIMENT ANALYSE - Aktiensuche
# ================================================================================
# Dieser Callback ermöglicht die Aktiensuche für die Sentiment-Analyse.
# Der Benutzer tippt einen Suchbegriff ein, und das Dropdown wird
# mit passenden Aktien gefüllt.
#
# Verwendet die globale search_stocks() Funktion, die Yahoo Finance abfragt.
# ================================================================================
@callback(
    Output("sentiment-stock-dropdown", "options"),   # Dropdown-Optionen
    Output("sentiment-stock-dropdown", "value"),     # Vorausgewählter Wert
    Input("sentiment-search-input", "value"),        # Suchbegriff vom Benutzer
    prevent_initial_call=True
)
def sentiment_search_stocks(search_query):
    """
    Sucht Aktien für die Sentiment-Analyse basierend auf der Benutzereingabe.
    
    Args:
        search_query: Der Suchbegriff (Aktienname oder Symbol)
    
    Returns:
        Tuple: (Dropdown-Optionen, vorausgewählter Wert)
    """
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

# ================================================================================
# SENTIMENT ANALYSE - Hauptanalyse-Callback
# ================================================================================
# Dieser Callback führt die eigentliche Sentiment-Analyse durch.
# Er analysiert News-Artikel zu einer Aktie und bewertet deren Stimmung.
#
# Technische Details:
# - Verwendet VADER (Valence Aware Dictionary and sEntiment Reasoner)
# - VADER ist ein regelbasierter Sentiment-Analysator für Social Media
# - Scores reichen von -1 (sehr negativ) bis +1 (sehr positiv)
#
# Die Analyse zeigt:
# - Durchschnittlicher Sentiment-Score
# - Anzahl analysierter News
# - Kursänderung im Zeitraum
# - Chart mit Sentiment über Zeit
# - Top News sortiert nach Sentiment-Stärke
# ================================================================================
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
    Führt eine Sentiment-Analyse für die ausgewählte Aktie durch.
    
    Args:
        n_clicks: Button-Klick-Zähler (zum Triggern)
        symbol: Das Aktien-Symbol (z.B. "AAPL")
        period: Analysezeitraum (z.B. "1mo", "3mo")
        news_count: Maximale Anzahl zu analysierender News
    
    Returns:
        html.Div: Formatierte Ergebnisanzeige mit Chart und Statistiken
    """
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

# ================================================================================
# KORRELATIONS-ANALYSE - Aktiensuche
# ================================================================================
# Aktiensuche für die Korrelationsanalyse.
# Funktioniert identisch zur Sentiment-Suche.
# ================================================================================
@callback(
    Output("corr-stock-dropdown", "options"),        # Dropdown-Optionen
    Output("corr-stock-dropdown", "value"),          # Vorausgewählter Wert
    Input("corr-search-input", "value"),             # Suchbegriff
    prevent_initial_call=True
)
def corr_search_stocks(search_query):
    """
    Sucht Aktien für die Korrelationsanalyse.
    """
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

# ================================================================================
# KORRELATIONS-ANALYSE - Hauptanalyse-Callback
# ================================================================================
# Dieser Callback berechnet die Korrelation zwischen News-Sentiment
# und Kursbewegung einer Aktie.
#
# Was ist Korrelation?
# - Ein statistisches Maß, das zeigt, wie stark zwei Variablen zusammenhängen
# - Werte von -1 bis +1:
#   * +1 = Perfekte positive Korrelation (wenn A steigt, steigt B)
#   * 0 = Keine lineare Beziehung
#   * -1 = Perfekte negative Korrelation (wenn A steigt, fällt B)
#
# Die Analyse zeigt:
# - Korrelationskoeffizient zwischen Sentiment und Kurs
# - Visualisierung der Beziehung im Chart
# - Interpretation der Ergebnisse
# ================================================================================
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
    Berechnet die Korrelation zwischen News-Sentiment und Kursbewegung.
    
    Args:
        n_clicks: Button-Klick-Zähler
        symbol: Das Aktien-Symbol
        period: Analysezeitraum
        news_count: Maximale Anzahl News
    
    Returns:
        html.Div: Formatierte Ergebnisanzeige mit Korrelations-Chart
    """
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


# ================================================================================
# PROGNOSE (FORECAST) CALLBACKS - ARIMA Zeitreihenanalyse
# ================================================================================
# Diese Callbacks implementieren die Kursprognose mittels ARIMA-Modell.
#
# Was ist ARIMA?
# ARIMA steht für: AutoRegressive Integrated Moving Average
# - AutoRegressive (AR): Nutzt vergangene Werte zur Vorhersage
# - Integrated (I): Macht die Daten stationär durch Differenzierung
# - Moving Average (MA): Nutzt vergangene Prognosefehler
#
# ARIMA-Parameter (p, d, q):
# - p: Anzahl der AR-Terme (Vergangenheitswerte)
# - d: Grad der Differenzierung (meist 1)
# - q: Anzahl der MA-Terme (Fehlerwerte)
#
# Wichtig: ARIMA ist ein statistisches Modell und keine Kristallkugel!
# Die Prognosen sind nur Schätzungen basierend auf historischen Mustern.
# ================================================================================

# --- Aktiensuche für Prognose ---
@callback(
    Output("forecast-stock-dropdown", "options"),    # Dropdown-Optionen
    Output("forecast-stock-dropdown", "value"),      # Vorausgewählter Wert
    Input("forecast-search-input", "value"),         # Suchbegriff
    prevent_initial_call=True
)
def forecast_search_callback(search_term):
    """
    Sucht Aktien für die Prognose-Funktion.
    
    Verwendet Yahoo Finance API direkt statt der globalen search_stocks(),
    um die Implementierung zu demonstrieren.
    """
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


# --- ARIMA-Prognose Hauptanalyse ---
# Dieser Callback führt die eigentliche ARIMA-Prognose durch
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
    Führt eine ARIMA-basierte Kursprognose durch.
    
    Args:
        n_clicks: Button-Klick-Zähler
        symbol: Das Aktien-Symbol
        history_period: Zeitraum der historischen Daten für Training
        forecast_days: Anzahl Tage für die Prognose
    
    Returns:
        html.Div: Prognose-Ergebnisse mit Chart und Konfidenzintervall
    
    Die Ergebnisse zeigen:
    - Aktueller Kurs vs. prognostizierter Kurs
    - 95% Konfidenzintervall (Unsicherheitsbereich)
    - ARIMA-Parameter und Modell-Güte (AIC-Score)
    """
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


# ================================================================================
# MONTE-CARLO SIMULATION CALLBACKS
# ================================================================================
# Monte-Carlo-Simulationen verwenden Zufallszahlen, um mögliche
# zukünftige Kursverläufe zu simulieren.
#
# Was ist Monte-Carlo?
# - Benannt nach dem Casino in Monaco
# - Verwendet Zufallszahlen um komplexe Systeme zu simulieren
# - Hier: Tausende mögliche Kursverläufe werden simuliert
#
# Geometric Brownian Motion (GBM):
# - Mathematisches Modell für Aktienkurse
# - Berücksichtigt: Drift (Trend) + Volatilität (Zufälligkeit)
# - Formel: dS = μ*S*dt + σ*S*dW
#   * μ (mu): Jährliche Drift/Rendite
#   * σ (sigma): Jährliche Volatilität
#   * dW: Zufallskomponente (Wiener-Prozess)
#
# Ergebnisse:
# - Wahrscheinlichkeitsverteilung möglicher Endpreise
# - Perzentile (5%, 25%, 50%, 75%, 95%)
# - Wahrscheinlichkeit für Gewinn/Verlust
# ================================================================================

# --- Aktiensuche für Monte-Carlo ---
@callback(
    Output("mc-stock-dropdown", "options"),          # Dropdown-Optionen
    Output("mc-stock-dropdown", "value"),            # Vorausgewählter Wert
    Input("mc-search-input", "value"),               # Suchbegriff
    prevent_initial_call=True
)
def mc_search_callback(search_term):
    """
    Sucht Aktien für die Monte-Carlo-Simulation.
    """
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
    Führt eine Monte-Carlo-Simulation für die Kursprognose durch.
    
    Args:
        n_clicks: Button-Klick-Zähler
        symbol: Das Aktien-Symbol
        history_period: Zeitraum für historische Volatilitätsberechnung
        forecast_days: Anzahl Tage für die Simulation
        num_simulations: Anzahl der Simulationspfade (mehr = genauer, aber langsamer)
    
    Returns:
        html.Div: Simulationsergebnisse mit:
        - Median-Prognose und Wahrscheinlichkeitsverteilung
        - Perzentile der möglichen Endpreise
        - Gewinn-/Verlust-Wahrscheinlichkeiten
        - Fächer-Chart mit allen Simulationspfaden
    """
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


# ================================================================================
# THEME TOGGLE - Clientside Callback für Dark/Light Mode
# ================================================================================
# Dieser Callback ist ein CLIENTSIDE Callback - er läuft direkt im Browser!
#
# Was ist ein Clientside Callback?
# - Normaler Callback: Browser -> Server -> Browser (langsam)
# - Clientside Callback: Läuft komplett im Browser (schnell)
# - Ideal für einfache UI-Änderungen wie Theme-Wechsel
#
# Der JavaScript-Code:
# 1. Prüft welcher Button geklickt wurde (triggered_id)
# 2. Fügt/entfernt die CSS-Klasse 'light-mode' am body
# 3. Speichert das aktuelle Theme im dcc.Store
#
# dash_clientside.callback_context: Entspricht ctx im Server-Callback
# dash_clientside.no_update: Verhindert unnötige Updates
# ================================================================================
app.clientside_callback(
    # JavaScript-Code als String
    # Dieser Code läuft direkt im Browser des Benutzers
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
    """,
    # Output: Speichert das aktuelle Theme
    Output("theme-store", "data"),
    # Inputs: Die Theme-Toggle-Buttons
    Input("btn-light-mode", "n_clicks"),
    Input("btn-dark-mode", "n_clicks"),
    prevent_initial_call=True
)


# ================================================================================
# SERVER STARTEN - Hier startet die Anwendung
# ================================================================================
# Der folgende Code wird nur ausgeführt, wenn dieses Skript direkt gestartet wird
# (nicht wenn es von einem anderen Skript importiert wird).
#
# __name__ == "__main__" ist ein Python-Idiom:
# - Wenn du das Skript direkt aufrufst: __name__ ist "__main__"
# - Wenn du es importierst: __name__ ist der Modulname
#
# app.run() startet den Dash-Entwicklungsserver:
# - debug=True: Zeigt detaillierte Fehlermeldungen und lädt bei Codeänderungen neu
# - port=8050: Die Anwendung ist unter http://localhost:8050 erreichbar
#
# Für Produktionseinsatz sollte ein richtiger WSGI-Server wie Gunicorn verwendet werden!
# ================================================================================

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
