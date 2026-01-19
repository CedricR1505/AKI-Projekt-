"""
================================================================================
SENTIMENT-ANALYSE MODUL FÜR DAS STOCK DASHBOARD
================================================================================

Dieses Modul enthält alle KI/ML-Analysefunktionen für das Stock Dashboard:

1. SENTIMENT-ANALYSE:
   - Sammelt Nachrichten aus verschiedenen RSS-Feeds
   - Analysiert die Stimmung (positiv/negativ) mit VADER
   - Korreliert Stimmung mit Kursbewegungen

2. ARIMA-PROGNOSE:
   - Zeitreihenanalyse mit ARIMA-Modell
   - Prognostiziert zukünftige Kurse
   - Berechnet Konfidenzintervalle

3. MONTE-CARLO-SIMULATION:
   - Simuliert tausende mögliche Kursverläufe
   - Basiert auf Geometric Brownian Motion (GBM)
   - Berechnet Wahrscheinlichkeiten für Gewinn/Verlust

HINWEIS FÜR ANFÄNGER:
- Dieses Modul wird von der Hauptanwendung (app_dash mit Kontostand.py) importiert
- Die Funktionen werden über Callbacks aufgerufen
- Du musst dieses Modul nicht direkt ausführen

AUTOR: Stock Dashboard Team
================================================================================
"""

# ================================================================================
# IMPORTS - Benötigte Bibliotheken
# ================================================================================

# requests: HTTP-Bibliothek zum Abrufen von RSS-Feeds aus dem Internet
import requests

# re: Regular Expressions (Reguläre Ausdrücke) zum Parsen von XML/HTML
# Wird verwendet um Titel, Daten und andere Infos aus RSS-Feeds zu extrahieren
import re

# pandas: DIE Python-Bibliothek für Datenanalyse
# - DataFrame: Tabellen-ähnliche Datenstruktur
# - Series: Eindimensionale Datenreihe
# - Viele Funktionen für Zeitreihen, Gruppierung, etc.
import pandas as pd

# numpy: Numerische Berechnungen und Arrays
# - Schnelle mathematische Operationen
# - Zufallszahlen für Monte-Carlo
# - Statistische Funktionen
import numpy as np

# yfinance: Yahoo Finance API Wrapper
# - Holt Aktienkurse und Unternehmensinformationen
# - Kostenlos und einfach zu benutzen
import yfinance as yf

# plotly.graph_objects: Interaktive Charts erstellen
# - Linien, Balken, Flächen, etc.
# - Hover-Effekte und Zoom
import plotly.graph_objects as go

# make_subplots: Mehrere Charts in einer Figur kombinieren
# - Ermöglicht Dual-Y-Achsen (z.B. Kurs + Sentiment)
from plotly.subplots import make_subplots

# datetime: Arbeiten mit Datum und Zeit
# - Datum parsen und formatieren
# - Zeiträume berechnen
from datetime import datetime, timedelta

# unescape: HTML-Entities dekodieren
# - Wandelt "&amp;" zurück zu "&"
# - Wichtig für saubere Titel aus RSS-Feeds
from html import unescape

# ================================================================================
# OPTIONALE BIBLIOTHEKEN - Mit Verfügbarkeitsprüfung
# ================================================================================
# Diese Bibliotheken sind optional. Falls sie nicht installiert sind,
# funktioniert die Anwendung trotzdem - nur ohne die entsprechenden Features.

# --- ARIMA für Zeitreihen-Prognose ---
# ARIMA = AutoRegressive Integrated Moving Average
# Ein statistisches Modell zur Vorhersage von Zeitreihen (wie Aktienkurse)
#
# Erklärung der Parameter:
# - AR (AutoRegressive): Nutzt vergangene Werte zur Vorhersage
# - I (Integrated): Differenziert die Daten für Stationarität
# - MA (Moving Average): Nutzt vergangene Fehler zur Korrektur
try:
    # ARIMA-Modell aus statsmodels
    from statsmodels.tsa.arima.model import ARIMA
    
    # Augmented Dickey-Fuller Test
    # Prüft ob eine Zeitreihe "stationär" ist (konstanter Mittelwert/Varianz)
    # Wichtig: ARIMA funktioniert am besten mit stationären Daten
    from statsmodels.tsa.stattools import adfuller
    
    # Warnungen unterdrücken (ARIMA gibt viele aus)
    import warnings
    warnings.filterwarnings('ignore')
    
    # Flag: ARIMA ist verfügbar
    ARIMA_AVAILABLE = True
except ImportError:
    # Falls statsmodels nicht installiert ist
    ARIMA_AVAILABLE = False

# --- VADER Sentiment Analyzer ---
# VADER = Valence Aware Dictionary and sEntiment Reasoner
# Ein regelbasierter Sentiment-Analysator, speziell für Social Media und News
#
# Wie VADER funktioniert:
# 1. Hat ein Wörterbuch mit ~7500 Wörtern und deren Sentiment-Scores
# 2. Berücksichtigt Verstärker ("sehr gut" > "gut")
# 3. Berücksichtigt Negation ("nicht gut" → negativ)
# 4. Erkennt Emoticons und Slang
#
# Output: compound Score von -1 (sehr negativ) bis +1 (sehr positiv)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    
    # Eine globale Instanz des Analyzers erstellen (Performance)
    # So muss nicht bei jedem Aufruf ein neuer Analyzer erstellt werden
    _analyzer = SentimentIntensityAnalyzer()
except ImportError:
    # Falls vaderSentiment nicht installiert ist
    VADER_AVAILABLE = False
    _analyzer = None


# ================================================================================
# KONSTANTEN - Konfigurationswerte für die Analyse
# ================================================================================

# PERIOD_DAYS_MAP: Wandelt Zeitraum-Strings in Tage um
# Wird verwendet um das "Cutoff-Datum" zu berechnen (ab wann News relevant sind)
#
# Beispiel: period="1mo" → 30 Tage zurück → nur News der letzten 30 Tage
PERIOD_DAYS_MAP = {
    "1d": 1,       # 1 Tag
    "5d": 7,       # 5 Tage (gerundet auf 1 Woche)
    "1mo": 30,     # 1 Monat
    "3mo": 90,     # 3 Monate
    "6mo": 180,    # 6 Monate
    "1y": 365,     # 1 Jahr
    "5y": 1825     # 5 Jahre (5 * 365)
}

# ================================================================================
# RSS-FEED TEMPLATES - Dynamische News-Quellen (mit Aktien-Symbol)
# ================================================================================
# Diese URLs werden mit dem Aktien-Symbol ergänzt ({symbol} wird ersetzt)
# Google News ist die Hauptquelle, da es viele Quellen aggregiert
#
# Warum mehrere Suchbegriffe?
# - "stock": Findet allgemeine Aktiennachrichten
# - "Aktie": Deutsche Nachrichten
# - "earnings": Quartalszahlen und Gewinnmeldungen
# - "CEO": Führungswechsel und Interviews
# - etc.
#
# Je mehr Quellen, desto mehr News = bessere Sentiment-Analyse
RSS_FEED_TEMPLATES = [
    # --- Google News (verschiedene Suchbegriffe) ---
    # Google News aggregiert News von tausenden Quellen weltweit
    
    # Allgemeine Aktien-Suche (Englisch)
    "https://news.google.com/rss/search?q={symbol}+stock&hl=en&gl=US&ceid=US:en",
    
    # Deutsche Aktien-Suche
    "https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=DE&ceid=DE:de",
    
    # Aktien-Anteile Suche
    "https://news.google.com/rss/search?q={symbol}+shares&hl=en&gl=US&ceid=US:en",
    
    # Investoren-News
    "https://news.google.com/rss/search?q={symbol}+investor&hl=en&gl=US&ceid=US:en",
    
    # Quartalszahlen und Gewinn
    "https://news.google.com/rss/search?q={symbol}+earnings&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+quarterly&hl=en&gl=US&ceid=US:en",
    
    # Führungskräfte-News
    "https://news.google.com/rss/search?q={symbol}+CEO&hl=en&gl=US&ceid=US:en",
    
    # Markt-Analysen
    "https://news.google.com/rss/search?q={symbol}+market&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={symbol}+analysis&hl=en&gl=US&ceid=US:en",
    
    # Kurs-News
    "https://news.google.com/rss/search?q={symbol}+price&hl=en&gl=US&ceid=US:en",
    
    # --- Yahoo Finance RSS ---
    # Direkter Feed von Yahoo Finance für das spezifische Symbol
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
]

# ================================================================================
# STATISCHE RSS-FEEDS - Allgemeine Finanznachrichten-Quellen
# ================================================================================
# Diese Feeds enthalten allgemeine Finanznachrichten (nicht symbol-spezifisch)
# Die News werden später nach dem Aktien-Symbol gefiltert
#
# Vorteil: Findet auch News, die das Symbol nicht im Titel haben
# Nachteil: Viele irrelevante News müssen gefiltert werden
STATIC_RSS_FEEDS = [
    # MarketWatch: Großes amerikanisches Finanzportal (Teil von Dow Jones)
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories/", "name": "MarketWatch"},
    
    # CNBC: Amerikanischer Finanznachrichtensender
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "name": "CNBC"},
    
    # MarketWatch Pulse: Schnelle Markt-Updates
    {"url": "https://feeds.marketwatch.com/marketwatch/marketpulse/", "name": "MarketWatch Pulse"},
    
    # Investing.com: Internationale Finanzplattform
    {"url": "https://www.investing.com/rss/news.rss", "name": "Investing.com"},
    
    # Seeking Alpha: Investment-Analysen und Meinungen
    {"url": "https://seekingalpha.com/market_currents.xml", "name": "Seeking Alpha"},
]


# ================================================================================
# HILFSFUNKTIONEN - Grundlegende Funktionen für die Analyse
# ================================================================================
# Diese Funktionen werden von den Hauptanalyse-Funktionen aufgerufen.
# Sie kümmern sich um einzelne, wiederverwendbare Aufgaben.


def get_cutoff_date(period: str):
    """
    Berechnet das Cutoff-Datum (Stichtag) basierend auf dem gewählten Zeitraum.
    
    Was ist ein Cutoff-Datum?
    - Alle News VOR diesem Datum werden ignoriert
    - Nur News AB diesem Datum werden analysiert
    
    Args:
        period: Zeitraum-String (z.B. "1mo", "3mo", "1y")
    
    Returns:
        datetime: Das Datum, ab dem News relevant sind
    
    Beispiel:
        >>> get_cutoff_date("1mo")  # Bei heute = 30.12.2025
        datetime(2025, 11, 30)      # Ergebnis: 30.11.2025
    """
    # Hole Anzahl Tage aus der Map (Standard: 30 Tage wenn nicht gefunden)
    days_back = PERIOD_DAYS_MAP.get(period, 30)
    
    # Berechne: Heute minus X Tage
    return datetime.now() - timedelta(days=days_back)


def identify_source(feed_url: str) -> str:
    """
    Identifiziert die Nachrichtenquelle anhand der URL.
    
    Wird verwendet um anzuzeigen, woher eine News stammt.
    
    Args:
        feed_url: Die URL des RSS-Feeds
    
    Returns:
        str: Name der Quelle (z.B. "Google News", "Yahoo Finance")
    
    Beispiel:
        >>> identify_source("https://news.google.com/rss/...")
        "Google News"
    """
    # Prüfe welche bekannte Domain in der URL vorkommt
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
    # Fallback für unbekannte Quellen
    return "Unbekannt"


def calculate_sentiment(text: str) -> float:
    """
    Berechnet den Sentiment-Score für einen Text mit VADER.
    
    Was ist Sentiment-Analyse?
    - Automatische Erkennung der "Stimmung" in einem Text
    - Positiv: "Apple reports record profits, stock soars!"
    - Negativ: "Tesla faces massive recall, shares plummet"
    - Neutral: "Company releases quarterly report"
    
    VADER Compound Score:
    - Kombiniert positive, negative und neutrale Scores
    - Bereich: -1.0 (extrem negativ) bis +1.0 (extrem positiv)
    - ~0: Neutral
    
    Args:
        text: Der zu analysierende Text (z.B. News-Titel)
    
    Returns:
        float: Sentiment-Score zwischen -1.0 und +1.0
               0.0 falls VADER nicht verfügbar
    
    Beispiel:
        >>> calculate_sentiment("Apple stock hits all-time high!")
        0.6369  # Positiv
        
        >>> calculate_sentiment("Company faces bankruptcy")
        -0.5423  # Negativ
    """
    # Sicherheitsprüfung: VADER muss verfügbar sein
    if not VADER_AVAILABLE or _analyzer is None:
        return 0.0
    
    # polarity_scores() gibt ein Dictionary zurück:
    # {"pos": 0.5, "neg": 0.0, "neu": 0.5, "compound": 0.6369}
    # Wir verwenden nur den "compound" Score (kombinierter Wert)
    return _analyzer.polarity_scores(text)["compound"]


def parse_date(date_str: str):
    """
    Versucht verschiedene Datumsformate zu parsen.
    
    Warum ist das kompliziert?
    - Verschiedene RSS-Feeds verwenden verschiedene Datumsformate!
    - Amerikanisch: "Mon, 30 Dec 2025 10:30:00 GMT"
    - ISO-Format: "2025-12-30T10:30:00Z"
    - Deutsch: "30.12.2025"
    - Mit/ohne Zeitzone
    
    Die Funktion probiert alle gängigen Formate durch, bis eines passt.
    
    Args:
        date_str: Datums-String in beliebigem Format
    
    Returns:
        datetime: Geparstes Datum (OHNE Zeitzone für einfachere Vergleiche)
                  Bei Fehler: Aktuelles Datum
    
    Hinweis: "timezone-naive" bedeutet ohne Zeitzone-Information.
    Das ist wichtig, weil man sonst keine Datums-Vergleiche machen kann!
    """
    # Leerer String? Gib aktuelles Datum zurück
    if not date_str:
        return datetime.now()
    
    # Liste aller unterstützten Datumsformate
    # Format-Codes erklärt:
    # %a = Wochentag kurz (Mon, Tue, ...)
    # %d = Tag (01-31)
    # %b = Monat kurz (Jan, Feb, ...)
    # %Y = Jahr 4-stellig (2025)
    # %H = Stunde (00-23)
    # %M = Minute (00-59)
    # %S = Sekunde (00-59)
    # %z = Zeitzone (+0100, -0500, etc.)
    # %Z = Zeitzone-Name (GMT, UTC, etc.)
    date_formats = [
        "%a, %d %b %Y %H:%M:%S %z",    # RFC 2822 mit Zeitzone
        "%a, %d %b %Y %H:%M:%S %Z",    # RFC 2822 mit TZ-Name
        "%a, %d %b %Y %H:%M:%S GMT",   # RFC 2822 GMT
        "%Y-%m-%dT%H:%M:%S%z",         # ISO 8601 mit TZ
        "%Y-%m-%dT%H:%M:%SZ",          # ISO 8601 UTC
        "%Y-%m-%d %H:%M:%S",           # Einfaches Format
        "%d %b %Y %H:%M:%S",           # Tag Monat Jahr
        "%Y-%m-%d",                     # Nur Datum
    ]
    
    result = None
    
    # Probiere jedes Format durch
    for fmt in date_formats:
        try:
            result = datetime.strptime(date_str.strip(), fmt)
            break  # Erfolg! Schleife verlassen
        except (ValueError, AttributeError):
            # Format passt nicht, probiere nächstes
            continue
    
    # Fallback: pandas ist sehr gut im Datum-Parsen
    # pandas probiert automatisch viele Formate durch
    if result is None:
        try:
            result = pd.to_datetime(date_str).to_pydatetime()
        except:
            # Nichts hat funktioniert → aktuelles Datum
            return datetime.now()
    
    # WICHTIG: Zeitzone entfernen (offset-naive machen)
    # Sonst bekommt man Fehler bei Vergleichen mit anderen Daten
    if result is not None and result.tzinfo is not None:
        result = result.replace(tzinfo=None)
    
    return result if result else datetime.now()


def fetch_rss_feed(url: str, timeout: int = 10):
    """
    Holt einen RSS-Feed aus dem Internet und extrahiert die Items.
    
    Was ist RSS?
    - RSS = Really Simple Syndication
    - Ein XML-Format für Nachrichtenfeeds
    - Ermöglicht das automatische Abrufen von News
    - Struktur: <channel> enthält mehrere <item> (=Nachrichten)
    
    Args:
        url: Die URL des RSS-Feeds
        timeout: Maximale Wartezeit in Sekunden (Standard: 10)
    
    Returns:
        list: Liste von XML-Strings, einer pro News-Item
              Leere Liste bei Fehler
    
    Hinweis: Die Funktion gibt nur die rohen XML-Strings zurück.
    Das Parsen der einzelnen Felder erfolgt in parse_feed_item().
    """
    try:
        # HTTP-Header setzen (wichtig für viele Server!)
        # Ohne User-Agent blocken manche Server die Anfrage
        headers = {
            # Browser-Identifikation (Chrome auf Windows)
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            
            # Akzeptierte Inhaltstypen (XML-Formate für RSS)
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            
            # Bevorzugte Sprachen
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        }
        
        # HTTP GET-Request senden
        resp = requests.get(url, timeout=timeout, headers=headers)
        
        # Status-Code prüfen (200 = OK)
        if resp.status_code != 200:
            return []
        
        # Response-Text (XML) holen
        content = resp.text
        
        # Items aus dem Feed extrahieren mit Regular Expression
        # Sucht nach allem zwischen <item> und </item>
        # re.DOTALL: . matcht auch Zeilenumbrüche
        # re.IGNORECASE: Groß-/Kleinschreibung egal
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL | re.IGNORECASE)
        
        # Alternativ: Atom-Feeds verwenden <entry> statt <item>
        if not items:
            items = re.findall(r"<entry>(.*?)</entry>", content, re.DOTALL | re.IGNORECASE)
        
        return items
        
    except Exception as e:
        # Fehler loggen (hilft beim Debugging)
        print(f"[RSS] Fehler bei {url}: {e}")
        return []


def parse_feed_item(item_xml: str, source_name: str) -> dict:
    """
    Parst ein einzelnes RSS-Item und extrahiert die relevanten Daten.
    
    Ein RSS-Item sieht typischerweise so aus:
    <item>
        <title>Apple announces new iPhone</title>
        <pubDate>Mon, 30 Dec 2025 10:30:00 GMT</pubDate>
        <source>TechCrunch</source>
        <description>Apple has announced...</description>
    </item>
    
    Args:
        item_xml: Der XML-String des Items (ohne äußere <item>-Tags)
        source_name: Name der Quelle (als Fallback)
    
    Returns:
        dict: Dictionary mit "title", "date" und "source"
    
    Hinweis: Diese Funktion verwendet Regular Expressions (Regex)
    zum Extrahieren der Daten. Ein XML-Parser wäre "sauberer",
    aber Regex ist schneller und robuster bei fehlerhaftem XML.
    """
    # === TITEL EXTRAHIEREN ===
    # Suche nach <title>...</title>
    title_m = re.search(r"<title[^>]*>(.*?)</title>", item_xml, re.DOTALL | re.IGNORECASE)
    
    if title_m:
        title = title_m.group(1)  # Inhalt der ersten Capture-Group
        
        # CDATA-Blöcke entfernen
        # CDATA wird verwendet um Sonderzeichen zu schützen: <![CDATA[Text]]>
        title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title, flags=re.DOTALL)
        
        # HTML-Entities dekodieren (&amp; → &, &lt; → <, etc.)
        title = unescape(title.strip())
        
        # Übrige HTML-Tags entfernen (z.B. <b>, <i>)
        title = re.sub(r"<[^>]+>", "", title)
    else:
        title = ""
    
    # === DATUM EXTRAHIEREN ===
    # Verschiedene Tags probieren (RSS vs. Atom haben unterschiedliche Namen)
    
    # RSS-Format: <pubDate>...</pubDate>
    pub_m = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item_xml, re.DOTALL | re.IGNORECASE)
    
    # Atom-Format: <published>...</published>
    if not pub_m:
        pub_m = re.search(r"<published[^>]*>(.*?)</published>", item_xml, re.DOTALL | re.IGNORECASE)
    
    # Atom-Format alternativ: <updated>...</updated>
    if not pub_m:
        pub_m = re.search(r"<updated[^>]*>(.*?)</updated>", item_xml, re.DOTALL | re.IGNORECASE)
    
    # Datum parsen (mit unserer flexiblen parse_date Funktion)
    pub_date = parse_date(pub_m.group(1) if pub_m else "")
    
    # === QUELLE EXTRAHIEREN ===
    # Manche Feeds haben eine <source>-Tag mit der Original-Quelle
    # (z.B. bei Google News, das von vielen Quellen aggregiert)
    source_m = re.search(r"<source[^>]*>(.*?)</source>", item_xml, re.DOTALL | re.IGNORECASE)
    
    if source_m:
        # Quelle bereinigen (CDATA und HTML-Entities)
        original_source = unescape(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", source_m.group(1)).strip())
    else:
        # Keine Quelle im Feed → Feed-Name verwenden
        original_source = source_name
    
    return {
        "title": title,
        "date": pub_date,
        "source": original_source
    }


def get_company_name(symbol: str) -> str:
    """
    Holt den Firmennamen von Yahoo Finance für ein Aktien-Symbol.
    
    Warum brauchen wir den Firmennamen?
    - Manche News erwähnen nicht das Symbol, sondern den Namen
    - "Tesla announces new factory" statt "TSLA announces..."
    - Mit dem Namen können wir mehr relevante News finden
    
    Args:
        symbol: Das Aktien-Symbol (z.B. "TSLA", "AAPL")
    
    Returns:
        str: Der Firmenname (z.B. "Tesla", "Apple")
             Leerer String bei Fehler
    
    Beispiel:
        >>> get_company_name("TSLA")
        "Tesla"
        >>> get_company_name("AAPL")
        "Apple"
    """
    try:
        # Yahoo Finance Ticker-Objekt erstellen
        ticker = yf.Ticker(symbol)
        
        # Info-Dictionary abrufen
        info = ticker.info
        
        # Name holen (shortName bevorzugt, sonst longName)
        name = info.get("shortName", "") or info.get("longName", "")
        
        if name:
            # Namen bereinigen:
            # "Tesla, Inc." → "Tesla"
            # "Apple Inc" → "Apple"
            # "Microsoft Corporation" → "Microsoft"
            return name.split(",")[0].split(" Inc")[0].split(" Corp")[0].strip()
    except:
        pass
    
    return ""


# ================================================================================
# NEWS-ABRUF - Nachrichten aus RSS-Feeds sammeln
# ================================================================================
# Diese Funktionen sammeln News aus vielen verschiedenen Quellen
# und bereiten sie für die Sentiment-Analyse vor.


def fetch_news_from_feeds(symbol: str, period: str = "1mo", news_limit: int = 100):
    """
    Sammelt News aus mehreren RSS-Feeds und berechnet Sentiment-Scores.
    
    Dies ist die HAUPT-FUNKTION für den News-Abruf!
    
    Ablauf:
    1. Cutoff-Datum berechnen (ab wann sind News relevant?)
    2. Firmennamen für bessere Suche holen
    3. Alle dynamischen Feeds durchgehen (mit Symbol)
    4. Alle statischen Feeds durchgehen (nach Symbol filtern)
    5. Duplikate entfernen
    6. Nach Datum sortieren
    7. Auf Limit begrenzen
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA", "AAPL", "MSFT")
        period: Zeitraum für die Analyse ("1mo", "3mo", "1y", etc.)
        news_limit: Maximale Anzahl der zurückgegebenen News
    
    Returns:
        tuple: (news_items, sources_found)
            - news_items: Liste von Dictionaries mit title, date, score, source
            - sources_found: Liste der gefundenen Quellen mit Anzahl
    
    Beispiel:
        >>> news, sources = fetch_news_from_feeds("TSLA", "1mo", 100)
        >>> print(f"Gefunden: {len(news)} News aus {sources}")
        Gefunden: 87 News aus ['Google News (72)', 'Yahoo Finance (15)']
    """
    # Stichtag berechnen: News vor diesem Datum werden ignoriert
    cutoff_date = get_cutoff_date(period)
    
    # Sammel-Listen
    all_news_items = []     # Alle gefundenen News
    sources_count = {}      # Zähler pro Quelle
    seen_titles = set()     # Set für Duplikat-Erkennung (schnell!)
    
    # Firmenname für bessere Filterung holen
    company_name = get_company_name(symbol)
    
    # Suchbegriffe erstellen (Symbol + Firmenname)
    search_terms = [symbol.upper()]
    if company_name:
        search_terms.append(company_name.upper())
    
    print(f"[Sentiment] Suche nach: {search_terms}")
    
    # =========================================================================
    # SCHRITT 1: Dynamische Feeds durchgehen (symbol-spezifisch)
    # =========================================================================
    # Diese Feeds werden mit dem Symbol ergänzt, z.B.:
    # "https://news.google.com/rss/search?q=TSLA+stock..."
    
    for template in RSS_FEED_TEMPLATES:
        # Symbol in die URL einsetzen
        url = template.format(symbol=symbol)
        source_name = identify_source(url)
        
        # Feed abrufen
        items = fetch_rss_feed(url)
        print(f"[RSS] {source_name}: {len(items)} Items gefunden")
        
        # Jedes Item verarbeiten
        for item_xml in items:
            # Item parsen (Titel, Datum, Quelle extrahieren)
            parsed = parse_feed_item(item_xml, source_name)
            
            # Zu kurze Titel überspringen (wahrscheinlich kein echter Artikel)
            if not parsed["title"] or len(parsed["title"]) < 10:
                continue
            
            # --- Duplikat-Check ---
            # Verwende die ersten 60 Zeichen des Titels als "Hash"
            # So werden "leicht unterschiedliche" Duplikate erkannt
            title_hash = parsed["title"].lower()[:60]
            if title_hash in seen_titles:
                continue  # Schon gesehen → überspringen
            seen_titles.add(title_hash)
            
            # --- Zeit-Filter ---
            # Nur News im gewählten Zeitraum akzeptieren
            try:
                item_date = parsed["date"]
                # Zeitzone entfernen für Vergleich
                if hasattr(item_date, 'tzinfo') and item_date.tzinfo is not None:
                    item_date = item_date.replace(tzinfo=None)
                if item_date < cutoff_date:
                    continue  # Zu alt → überspringen
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
    
    # =========================================================================
    # SCHRITT 2: Statische Feeds durchgehen (allgemeine Finanznews)
    # =========================================================================
    # Diese Feeds enthalten ALLE Finanznews, nicht nur für unser Symbol.
    # Deshalb müssen wir nach dem Symbol/Firmennamen filtern.
    
    for feed in STATIC_RSS_FEEDS:
        # Feed abrufen (URL ist fest, nicht dynamisch)
        items = fetch_rss_feed(feed["url"])
        print(f"[RSS] {feed['name']}: {len(items)} Items gefunden")
        
        for item_xml in items:
            parsed = parse_feed_item(item_xml, feed["name"])
            
            # Leere Titel überspringen
            if not parsed["title"]:
                continue
            
            # --- WICHTIG: Symbol-Filter ---
            # Prüfen ob Symbol oder Firmenname im Titel vorkommt
            # Sonst ist die News nicht relevant für uns!
            title_upper = parsed["title"].upper()
            if not any(term in title_upper for term in search_terms):
                continue  # Nicht relevant → überspringen
            
            # Duplikat-Check (wie oben)
            title_hash = parsed["title"].lower()[:60]
            if title_hash in seen_titles:
                continue
            seen_titles.add(title_hash)
            
            # Zeit-Filter (wie oben)
            try:
                item_date = parsed["date"]
                if hasattr(item_date, 'tzinfo') and item_date.tzinfo is not None:
                    item_date = item_date.replace(tzinfo=None)
                if item_date < cutoff_date:
                    continue
            except Exception:
                pass
            
            # Sentiment berechnen
            score = calculate_sentiment(parsed["title"])
            
            # Zur Liste hinzufügen
            all_news_items.append({
                "title": parsed["title"],
                "date": parsed["date"].strftime("%d.%m.%Y"),
                "date_obj": parsed["date"],
                "score": score,
                "source": feed["name"],
                "feed_source": feed["name"]
            })
            
            sources_count[feed["name"]] = sources_count.get(feed["name"], 0) + 1
    
    # =========================================================================
    # SCHRITT 3: Sortieren, Begrenzen und Bereinigen
    # =========================================================================
    
    # Nach Datum sortieren (neueste zuerst)
    # key=lambda x: x["date_obj"] → Sortiere nach dem date_obj Feld
    # reverse=True → Absteigend (neueste zuerst)
    all_news_items.sort(key=lambda x: x["date_obj"], reverse=True)
    
    # Auf das Limit begrenzen (z.B. nur die 100 neuesten)
    news_items = all_news_items[:news_limit]
    
    # Temporäre Felder entfernen (werden für die Rückgabe nicht benötigt)
    # und finale Quellen-Statistik erstellen
    final_sources = {}
    for item in news_items:
        # Quellen zählen
        feed_src = item.get("feed_source", item["source"])
        final_sources[feed_src] = final_sources.get(feed_src, 0) + 1
        
        # Interne Felder löschen
        if "date_obj" in item:
            del item["date_obj"]
        if "feed_source" in item:
            del item["feed_source"]
    
    # Quellen-Zusammenfassung für die Anzeige erstellen
    # Format: ["Google News (72)", "Yahoo Finance (15)", ...]
    sources_found = [f"{name} ({count})" for name, count in sorted(final_sources.items(), key=lambda x: -x[1])]
    
    print(f"[Sentiment] Insgesamt {len(news_items)} News (von {len(all_news_items)} gefunden) aus: {sources_found}")
    
    return news_items, sources_found


def fetch_news_for_correlation(symbol: str, period: str = "3mo"):
    """
    Holt News speziell für die Korrelationsanalyse.
    
    Wrapper um fetch_news_from_feeds() mit angepassten Parametern:
    - Längerer Standardzeitraum (3 Monate statt 1 Monat)
    - Mehr News (500 statt 100) für bessere statistische Aussagekraft
    
    Args:
        symbol: Aktiensymbol
        period: Zeitraum (Standard: 3 Monate)
    
    Returns:
        list: Liste der News-Items (ohne Quellen-Info)
    """
    news_items, _ = fetch_news_from_feeds(symbol, period, 500)
    return news_items


# ================================================================================
# CHART-ERSTELLUNG - Visualisierungen für die Analyse
# ================================================================================
# Diese Funktionen erstellen interaktive Plotly-Charts für die Anzeige
# im Dashboard. Alle Charts sind responsive und haben Hover-Effekte.


def create_sentiment_chart(symbol: str, hist, sentiment_daily) -> go.Figure:
    """
    Erstellt einen Dual-Axis Chart mit Kurs und Sentiment.
    
    Was ist ein Dual-Axis Chart?
    - Ein Chart mit ZWEI Y-Achsen
    - Links: Aktienkurs (in USD)
    - Rechts: Sentiment-Score (-1 bis +1)
    - So können wir beide Werte in einem Chart vergleichen
    
    Args:
        symbol: Aktiensymbol für den Titel
        hist: DataFrame mit historischen Kursdaten (von yfinance)
        sentiment_daily: Series mit täglichen Sentiment-Durchschnitten
    
    Returns:
        go.Figure: Plotly-Figur bereit zur Anzeige
    """
    # make_subplots mit secondary_y=True ermöglicht zweite Y-Achse
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Farbe basierend auf Kursentwicklung wählen
    start_price = hist["Close"].iloc[0]   # Erster Kurs
    end_price = hist["Close"].iloc[-1]     # Letzter Kurs
    is_positive = end_price >= start_price # Ist der Kurs gestiegen?
    color_line = "#22c55e" if is_positive else "#ef4444"  # Grün oder Rot
    
    # === KURSLINIE HINZUFÜGEN ===
    # go.Scatter erstellt eine Linie (oder Punkte)
    # secondary_y=False → Linke Y-Achse
    fig.add_trace(
        go.Scatter(
            x=hist.index,               # X-Achse: Datums-Index
            y=hist["Close"],            # Y-Achse: Schlusskurse
            mode="lines",               # Nur Linie, keine Punkte
            name=f"{symbol} Kurs",      # Name für die Legende
            line=dict(color=color_line, width=2),  # Linien-Stil
            hovertemplate="%{y:.2f} USD<extra></extra>"  # Hover-Text
        ),
        secondary_y=False  # Linke Y-Achse
    )
    
    # === SENTIMENT-BALKEN HINZUFÜGEN ===
    # Nur wenn Sentiment-Daten vorhanden sind
    if len(sentiment_daily) > 0:
        # Farbe für jeden Balken: Grün wenn positiv, Rot wenn negativ
        colors_bars = ["#22c55e" if s >= 0 else "#ef4444" for s in sentiment_daily.values]
        
        # go.Bar erstellt Balken-Chart
        fig.add_trace(
            go.Bar(
                x=pd.to_datetime(sentiment_daily.index),  # X-Achse: Datum
                y=sentiment_daily.values,                  # Y-Achse: Sentiment-Score
                name="Sentiment Score",                    # Legende
                marker_color=colors_bars,                  # Individuelle Balken-Farben
                opacity=0.5,                               # Halbtransparent
                hovertemplate="Sentiment: %{y:.2f}<extra></extra>"
            ),
            secondary_y=True  # Rechte Y-Achse
        )
    
    # === LAYOUT KONFIGURIEREN ===
    # Prozentuale Änderung für den Titel berechnen
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
    """
    Erstellt einen Korrelations-Chart mit Kurs und Sentiment-Overlay.
    
    Dieser Chart zeigt die Beziehung zwischen Sentiment und Kurs:
    - Oberer Teil: Kursverlauf mit farbiger Hintergrund-Markierung
      (Grün = positives Sentiment, Rot = negatives Sentiment)
    - Unterer Teil: Sentiment-Score als Balken (7-Tage-Durchschnitt)
    
    Args:
        symbol: Aktiensymbol für den Titel
        merged_df: DataFrame mit Spalten: date, price, sentiment, sentiment_ma
    
    Returns:
        go.Figure: Plotly-Figur mit zwei übereinander liegenden Charts
    """
    # Zwei Zeilen, eine Spalte: Oben größer (70%), unten kleiner (30%)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],       # Größenverhältnis
        shared_xaxes=True,             # Gleiche X-Achse für beide
        vertical_spacing=0.05,         # Abstand zwischen Charts
        subplot_titles=(f"{symbol} Kurs mit Sentiment-Overlay", "Sentiment-Score (7-Tage Durchschnitt)")
    )
    
    # Farbe basierend auf Kursentwicklung
    start_price = merged_df["price"].iloc[0]
    end_price = merged_df["price"].iloc[-1]
    is_positive = end_price >= start_price
    color_price = "#22c55e" if is_positive else "#ef4444"
    
    # === OBERER CHART: Kurslinie ===
    fig.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["price"],
            mode="lines",
            name="Kurs",
            line=dict(color=color_price, width=2),
            hovertemplate="%{y:.2f} USD<extra>Kurs</extra>"
        ),
        row=1, col=1  # Position: Zeile 1, Spalte 1
    )
    
    # === SENTIMENT-HINTERGRUND ===
    # Färbt den Hintergrund je nach Sentiment ein
    # add_vrect = vertikales Rechteck (von oben bis unten)
    for i in range(len(merged_df) - 1):
        sentiment_val = merged_df["sentiment"].iloc[i]
        
        # Nur bei ausreichend starkem Sentiment färben
        if abs(sentiment_val) > 0.1:
            # Farbe wählen (halbtransparent mit rgba)
            fill_color = "rgba(34, 197, 94, 0.15)" if sentiment_val > 0 else "rgba(239, 68, 68, 0.15)"
            
            fig.add_vrect(
                x0=merged_df["date"].iloc[i],      # Start-Datum
                x1=merged_df["date"].iloc[i+1],    # End-Datum
                fillcolor=fill_color,
                layer="below",                      # Hinter der Linie
                line_width=0,                       # Kein Rahmen
                row=1, col=1
            )
    
    # === UNTERER CHART: Sentiment-Balken ===
    # Zeigt den 7-Tage-Durchschnitt des Sentiments
    colors_bars = ["#22c55e" if s > 0 else "#ef4444" for s in merged_df["sentiment_ma"]]
    fig.add_trace(
        go.Bar(
            x=merged_df["date"],
            y=merged_df["sentiment_ma"],
            name="Sentiment (MA7)",   # MA7 = Moving Average 7 Tage
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


# ================================================================================
# ANALYSE-FUNKTIONEN - Hauptfunktionen für die Sentiment-Analyse
# ================================================================================
# Diese Funktionen werden direkt vom Dashboard aufgerufen.
# Sie orchestrieren den gesamten Analyse-Prozess.


def analyze_sentiment(symbol: str, period: str = "1mo", news_limit: int = 100) -> dict:
    """
    Führt eine VOLLSTÄNDIGE Sentiment-Analyse durch.
    
    Dies ist die HAUPT-ANALYSE-FUNKTION, die vom Dashboard aufgerufen wird!
    
    Ablauf:
    1. Prüfen ob VADER verfügbar ist
    2. News aus RSS-Feeds abrufen
    3. Sentiment-Scores berechnen
    4. Tägliche Durchschnitte bilden
    5. Kursdaten von Yahoo Finance holen
    6. Chart erstellen
    7. Statistiken berechnen
    8. Alles als Dictionary zurückgeben
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA", "AAPL")
        period: Zeitraum für die Analyse ("1mo", "3mo", "1y", etc.)
        news_limit: Maximale Anzahl der News zu analysieren
    
    Returns:
        dict: Bei Erfolg:
              {
                  "success": True,
                  "symbol": "TSLA",
                  "news_items": [...],       # Liste der News
                  "sources_found": [...],    # Gefundene Quellen
                  "sentiment_daily": ...,    # Tägliche Scores
                  "figure": ...,             # Plotly Chart
                  "stats": {...}             # Statistiken
              }
              Bei Fehler:
              {"error": "Fehlermeldung"}
    """
    # Sicherheitsprüfung: VADER muss installiert sein
    if not VADER_AVAILABLE:
        return {"error": "vaderSentiment nicht installiert. Installieren mit: pip install vaderSentiment"}
    
    # Symbol bereinigen (Leerzeichen entfernen, Großbuchstaben)
    symbol = symbol.strip().upper()
    
    try:
        # === SCHRITT 1: News abrufen ===
        news_items, sources_found = fetch_news_from_feeds(symbol, period, news_limit)
        
        # Keine News gefunden?
        if not news_items:
            return {"error": f"Keine News für '{symbol}' gefunden. Versuchen Sie einen längeren Zeitraum."}
        
        # === SCHRITT 2: DataFrame erstellen ===
        # pandas DataFrame für einfachere Datenverarbeitung
        news_df = pd.DataFrame(news_items)
        
        # Datum-Spalte in datetime konvertieren
        news_df["date_parsed"] = pd.to_datetime(news_df["date"], format="%d.%m.%Y", errors="coerce")
        
        # === SCHRITT 3: Täglicher Sentiment-Durchschnitt ===
        # groupby: Gruppiert alle News nach Tag
        # mean(): Berechnet den Durchschnitt der Scores pro Tag
        sentiment_daily = news_df.groupby(news_df["date_parsed"].dt.date)["score"].mean()
        
        # === SCHRITT 4: Kursdaten abrufen ===
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period or "1mo")
        
        # Keine Kursdaten?
        if hist.empty:
            return {"error": f"Keine Kursdaten für '{symbol}' verfügbar."}
        
        # === SCHRITT 5: Chart erstellen ===
        fig = create_sentiment_chart(symbol, hist, sentiment_daily)
        
        # === SCHRITT 6: Statistiken berechnen ===
        start_price = hist["Close"].iloc[0]   # Erster Kurs im Zeitraum
        end_price = hist["Close"].iloc[-1]     # Letzter Kurs im Zeitraum
        
        # Prozentuale Kursänderung
        pct_change = ((end_price - start_price) / start_price) * 100
        
        # Durchschnittlicher Sentiment-Score aller News
        avg_sentiment = news_df["score"].mean()
        
        # === SCHRITT 7: Ergebnis zurückgeben ===
        return {
            "success": True,
            "symbol": symbol,
            "news_items": news_items,          # Alle analysierten News
            "sources_found": sources_found,    # Liste der Quellen
            "sentiment_daily": sentiment_daily,# Tägliche Sentiment-Scores
            "figure": fig,                     # Plotly-Chart
            "stats": {
                "avg_sentiment": avg_sentiment,     # Durchschnittliches Sentiment
                "news_count": len(news_items),      # Anzahl News
                "sentiment_days": len(sentiment_daily),  # Tage mit News
                "start_price": start_price,         # Anfangskurs
                "end_price": end_price,             # Endkurs
                "pct_change": pct_change,           # Prozentuale Änderung
                "is_positive": end_price >= start_price,  # Kurs gestiegen?
            }
        }
        
    except Exception as e:
        # Fehler abfangen und zurückgeben
        return {"error": str(e)}


def analyze_correlation(symbol: str, period: str = "3mo", news_limit: int = 500) -> dict:
    """
    Führt eine KORRELATIONSANALYSE zwischen Kurs und Sentiment durch.
    
    Was ist Korrelation?
    - Ein statistisches Maß für den Zusammenhang zweier Variablen
    - Werte von -1 bis +1:
      * +1: Perfekt positiv (wenn A steigt, steigt B immer)
      * 0: Kein linearer Zusammenhang
      * -1: Perfekt negativ (wenn A steigt, fällt B immer)
    
    Anwendung hier:
    - Wir messen: "Beeinflusst positives Sentiment den Aktienkurs?"
    - Hohe positive Korrelation = Gutes Sentiment → Kurs steigt
    - Niedrige/keine Korrelation = Sentiment hat wenig Einfluss
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        period: Zeitraum für die Analyse (Standard: 3 Monate)
        news_limit: Maximale Anzahl der News (mehr = bessere Statistik)
    
    Returns:
        dict: Ergebnis mit Korrelationskoeffizient und Chart
    """
    # VADER prüfen
    if not VADER_AVAILABLE:
        return {"error": "vaderSentiment nicht installiert"}
    
    symbol = symbol.strip().upper()
    days_back = PERIOD_DAYS_MAP.get(period, 90)
    
    try:
        # === SCHRITT 1: News abrufen ===
        news_items, _ = fetch_news_from_feeds(symbol, period, news_limit)
        
        # Mindestens 5 News für sinnvolle Korrelation
        if len(news_items) < 5:
            return {"error": f"Zu wenige News für '{symbol}' gefunden ({len(news_items)} Artikel). Versuchen Sie einen längeren Zeitraum."}
        
        # === SCHRITT 2: DataFrame erstellen ===
        news_df = pd.DataFrame(news_items)
        news_df["date_parsed"] = pd.to_datetime(news_df["date"], format="%d.%m.%Y", errors="coerce")
        
        # === SCHRITT 3: Täglicher Sentiment-Durchschnitt ===
        sentiment_daily = news_df.groupby(news_df["date_parsed"].dt.date)["score"].mean().reset_index()
        sentiment_daily.columns = ["date", "sentiment"]  # Spalten umbenennen
        sentiment_daily["date"] = pd.to_datetime(sentiment_daily["date"])
        
        # === SCHRITT 4: Kursdaten abrufen ===
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period or "3mo")
        
        if hist.empty:
            return {"error": f"Keine Kursdaten für '{symbol}' verfügbar."}
        
        # Kursdaten vorbereiten (Index zu Spalte machen)
        price_df = hist[["Close"]].reset_index()
        price_df.columns = ["date", "price"]
        # Zeitzone entfernen für Merge
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)
        
        # === SCHRITT 5: Daten zusammenführen (Merge) ===
        # pd.merge verbindet die DataFrames basierend auf dem Datum
        # how="left": Alle Kursdaten behalten, auch wenn kein Sentiment
        merged_df = pd.merge(price_df, sentiment_daily, on="date", how="left")
        
        # Fehlende Sentiment-Werte interpolieren (linear auffüllen)
        merged_df["sentiment"] = merged_df["sentiment"].interpolate(method="linear").fillna(0)
        
        # === SCHRITT 6: Korrelation berechnen ===
        # .corr() berechnet den Pearson-Korrelationskoeffizienten
        correlation = merged_df["price"].corr(merged_df["sentiment"])
        if pd.isna(correlation):  # Falls nicht berechenbar
            correlation = 0.0
        
        # === SCHRITT 7: Glättung mit Rolling Average ===
        # 7-Tage-Durchschnitt für glättere Darstellung
        # min_periods=1: Auch bei weniger als 7 Tagen berechnen
        merged_df["sentiment_ma"] = merged_df["sentiment"].rolling(window=7, min_periods=1).mean()
        
        # === SCHRITT 8: Chart erstellen ===
        fig = create_correlation_chart(symbol, merged_df)
        
        # === SCHRITT 9: Statistiken berechnen ===
        start_price = merged_df["price"].iloc[0]
        end_price = merged_df["price"].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        avg_sentiment = merged_df["sentiment"].mean()
        
        # === SCHRITT 10: Ergebnis zurückgeben ===
        return {
            "success": True,
            "symbol": symbol,
            "correlation": correlation,       # Der Korrelationskoeffizient!
            "figure": fig,                    # Plotly-Chart
            "merged_df": merged_df,           # Zusammengeführte Daten
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
    """
    Gibt ein Label und eine Farbe für einen Sentiment-Score zurück.
    
    Wird verwendet um den Score benutzerfreundlich anzuzeigen.
    
    Args:
        score: Sentiment-Score (-1 bis +1)
    
    Returns:
        tuple: (Label-Text, Bootstrap-Farbe)
               z.B. ("positiv", "success") oder ("negativ", "danger")
    
    Schwellenwerte:
    - > 0.05: Positiv (grün)
    - < -0.05: Negativ (rot)
    - Dazwischen: Neutral (grau)
    """
    if score > 0.05:
        return "positiv", "success"    # Grün
    elif score < -0.05:
        return "negativ", "danger"     # Rot
    return "neutral", "secondary"      # Grau


def get_correlation_label(correlation: float) -> tuple:
    """
    Gibt ein Label und eine Farbe für einen Korrelationskoeffizienten zurück.
    
    Interpretiert die Korrelation für den Benutzer.
    
    Args:
        correlation: Korrelationskoeffizient (-1 bis +1)
    
    Returns:
        tuple: (Label-Text, Bootstrap-Farbe)
    
    Schwellenwerte (nach statistischer Konvention):
    - > 0.5: Starke positive Korrelation
    - 0.3 bis 0.5: Moderate positive Korrelation
    - -0.3 bis 0.3: Schwache/keine Korrelation
    - -0.5 bis -0.3: Moderate negative Korrelation
    - < -0.5: Starke negative Korrelation
    """
    if correlation > 0.5:
        return "Stark positiv", "success"
    elif correlation > 0.3:
        return "Positiv", "success"
    elif correlation < -0.5:
        return "Stark negativ", "danger"
    elif correlation < -0.3:
        return "Negativ", "danger"
    return "Schwach/Neutral", "secondary"


# ================================================================================
# ARIMA PROGNOSE - Zeitreihenanalyse für Kursprognosen
# ================================================================================
#
# Was ist ARIMA?
# ARIMA = AutoRegressive Integrated Moving Average
# Ein statistisches Modell zur Vorhersage von Zeitreihen.
#
# Die drei Komponenten (p, d, q):
# - AR (p): AutoRegressive - nutzt vergangene Werte
#   "Der Kurs von morgen hängt vom Kurs von heute ab"
#
# - I (d): Integrated - Differenzierung für Stationarität
#   "Wir schauen auf ÄNDERUNGEN statt absolute Werte"
#
# - MA (q): Moving Average - nutzt vergangene Fehler
#   "Wenn wir gestern daneben lagen, korrigieren wir heute"
#
# Wichtig: ARIMA ist KEINE Kristallkugel!
# Es erkennt nur Muster in historischen Daten und extrapoliert diese.
# Unvorhersehbare Ereignisse (Kriege, Skandale, etc.) kann es nicht vorhersagen.
#
# ================================================================================


def analyze_forecast(symbol: str, history_period: str = "1y", forecast_days: int = 30) -> dict:
    """
    Führt eine ARIMA-basierte Kursprognose mit Trend-Korrektur durch.
    
    Diese Funktion ist komplex, weil sie mehrere Probleme löst:
    1. ARIMA neigt zu "Mean Reversion" (Rückkehr zum Mittelwert)
    2. Bei langen Prognosen wird das zu pessimistisch/optimistisch
    3. Deshalb: Kombination aus ARIMA + historischem Trend
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        history_period: Zeitraum für Training ("1y", "2y", etc.)
        forecast_days: Anzahl Tage für die Prognose
    
    Returns:
        dict: Bei Erfolg: Chart, Statistiken, Konfidenzintervalle
              Bei Fehler: {"error": "Fehlermeldung"}
    """
    # ARIMA prüfen
    if not ARIMA_AVAILABLE:
        return {"error": "statsmodels nicht installiert. Installieren mit: pip install statsmodels"}
    
    symbol = symbol.strip().upper()
    
    try:
        # === SCHRITT 1: Kursdaten abrufen ===
        stock = yf.Ticker(symbol)
        hist = stock.history(period=history_period)
        
        # Mindestens 30 Datenpunkte für sinnvolle Prognose
        if hist.empty or len(hist) < 30:
            return {"error": f"Nicht genügend Kursdaten für '{symbol}'. Mindestens 30 Datenpunkte benötigt."}
        
        # === SCHRITT 2: Daten vorbereiten ===
        # Wir extrahieren nur die Schlusskurse als NumPy-Array
        close_prices = hist["Close"].values
        dates_original = hist.index.tolist()
        
        # Zeitzonen entfernen (führt sonst zu Problemen)
        dates_clean = []
        for d in dates_original:
            if hasattr(d, 'to_pydatetime'):
                dt = d.to_pydatetime()
            else:
                dt = pd.Timestamp(d).to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            dates_clean.append(dt)
        
        # DataFrame erstellen (für spätere Chart-Erstellung)
        df = pd.DataFrame({
            "Close": close_prices,
            "Date": dates_clean
        })
        df = df.dropna()  # Leere Werte entfernen
        
        if len(df) < 30:
            return {"error": f"Nach Bereinigung nicht genügend Daten für '{symbol}'."}
        
        series = df["Close"].values  # Reine Werte für ARIMA
        
        # =================================================================
        # SCHRITT 3: Historischen Trend (Drift) berechnen
        # =================================================================
        # Log-Renditen sind stabiler als prozentuale Renditen
        # log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
        log_returns = np.diff(np.log(series))
        
        # Durchschnittliche tägliche Log-Rendite = "Drift"
        daily_drift = np.mean(log_returns)
        
        # Standardabweichung = "Volatilität"
        daily_volatility = np.std(log_returns)
        
        # Annualisierte Werte (252 Handelstage pro Jahr)
        annual_drift = daily_drift * 252
        annual_volatility = daily_volatility * np.sqrt(252)  # Wurzel wegen Varianz
        
        # =================================================================
        # SCHRITT 4: ARIMA-Modell anpassen
        # =================================================================
        
        # Stationarität prüfen mit Augmented Dickey-Fuller Test
        # p-value > 0.05 → Daten sind NICHT stationär → d=1
        d = 0  # Differenzierungsgrad
        try:
            adf_result = adfuller(series, autolag='AIC')
            if adf_result[1] > 0.05:  # p-value
                d = 1  # Einmal differenzieren
        except:
            d = 1  # Im Zweifel differenzieren
        
        # ARIMA-Modell mit Trend (trend='t' für linearen Trend bei d>0)
        # Grid-Search für beste ARIMA-Parameter
        # Wir probieren verschiedene (p, d, q) Kombinationen
        # und wählen die mit dem niedrigsten AIC (Akaike Information Criterion)
        best_aic = float('inf')  # Unendlich als Startwert
        best_order = (1, d, 1)   # Fallback-Parameter
        best_model = None
        
        # Verschiedene ARIMA-Parameter testen (p: 1-3, q: 1-2)
        for p in [1, 2, 3]:  # AR-Terme
            for q in [1, 2]:  # MA-Terme
                try:
                    # Trend-Parameter wählen:
                    # - Bei d>0: 't' (linear) erlaubt, 'c' (konstant) nicht
                    # - Bei d=0: 'c' (konstant) erlaubt
                    trend_param = 't' if d > 0 else 'c'
                    
                    # Modell erstellen und anpassen
                    model = ARIMA(series, order=(p, d, q), trend=trend_param)
                    fitted = model.fit()
                    
                    # Ist dieses Modell besser? (niedrigerer AIC = besser)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    # Manche Kombinationen funktionieren nicht → ignorieren
                    continue
        
        # Fallback falls keine Kombination funktioniert hat
        if best_model is None:
            model = ARIMA(series, order=(1, 1, 1))
            best_model = model.fit()
            best_order = (1, 1, 1)
        
        # =================================================================
        # SCHRITT 5: Prognose erstellen (mit Trend-Korrektur)
        # =================================================================
        
        # Basis-ARIMA-Prognose
        try:
            arima_forecast = best_model.forecast(steps=forecast_days)
            if hasattr(arima_forecast, 'values'):
                arima_forecast = arima_forecast.values
        except:
            arima_forecast = best_model.predict(start=len(series), end=len(series) + forecast_days - 1)
            if hasattr(arima_forecast, 'values'):
                arima_forecast = arima_forecast.values
        
        # --- Trend-Korrektur für lange Prognosen ---
        # Problem: ARIMA neigt zu "Mean Reversion" (Rückkehr zum Mittelwert)
        # Bei langen Prognosen ignoriert es den langfristigen Trend!
        # Lösung: Kombiniere ARIMA mit dem historischen Trend
        
        current_price = series[-1]  # Aktueller Kurs
        
        # Nur bei Prognosen > 90 Tage korrigieren
        if forecast_days > 90:
            # Gewichtung: Je länger die Prognose, desto mehr Trend
            # Max 70% Trend-Gewicht bei 5 Jahren
            trend_weight = min(0.7, forecast_days / 1825)
            
            # Trend-basierte Prognose mit exponentiellem Wachstum
            trend_forecast = np.zeros(forecast_days)
            trend_forecast[0] = current_price
            for t in range(1, forecast_days):
                # Exponentielles Wachstum: P(t) = P(t-1) * e^drift
                trend_forecast[t] = trend_forecast[t-1] * np.exp(daily_drift)
            
            # Kombiniere ARIMA und Trend (gewichteter Durchschnitt)
            forecast_mean = (1 - trend_weight) * arima_forecast + trend_weight * trend_forecast
        else:
            # Kurze Prognosen: Nur ARIMA verwenden
            forecast_mean = arima_forecast
        
        # =================================================================
        # SCHRITT 6: Konfidenzintervall berechnen
        # =================================================================
        # Das Konfidenzintervall zeigt die Unsicherheit der Prognose
        # 95% CI = "Mit 95% Wahrscheinlichkeit liegt der Kurs in diesem Bereich"
        
        residuals = best_model.resid  # Fehler des Modells
        std_err = np.std(residuals)   # Standardfehler
        
        # Unsicherheit wächst mit der Zeit (logarithmisch für Stabilität)
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
    """
    Gibt Label und Farbe für eine Prognose-Änderung zurück.
    
    Interpretiert die prognostizierte Kursänderung für den Benutzer.
    
    Args:
        change: Prognostizierte Änderung in Prozent
    
    Returns:
        tuple: (Label-Text, Bootstrap-Farbe)
    """
    if change > 5:
        return "Stark steigend", "success"
    elif change > 2:
        return "Steigend", "success"
    elif change < -5:
        return "Stark fallend", "danger"
    elif change < -2:
        return "Fallend", "danger"
    return "Seitwärts", "secondary"


# ================================================================================
# MONTE-CARLO SIMULATION - Stochastische Kursprognose
# ================================================================================
#
# Was ist Monte-Carlo?
# - Benannt nach dem berühmten Casino in Monaco
# - Methode: Tausende Zufallsexperimente durchführen
# - Aus den Ergebnissen Wahrscheinlichkeiten ableiten
#
# Anwendung bei Aktienkursen:
# - Wir simulieren TAUSENDE mögliche Kursverläufe
# - Jeder Verlauf ist zufällig, aber statistisch plausibel
# - Am Ende: Wahrscheinlichkeitsverteilung möglicher Kurse
#
# Geometric Brownian Motion (GBM):
# Das mathematische Modell hinter der Simulation:
# dS = μ*S*dt + σ*S*dW
#
# Wobei:
# - S: Aktienkurs
# - μ (mu): Drift = erwartete Rendite (Trend)
# - σ (sigma): Volatilität = Schwankungsstärke
# - dW: Wiener-Prozess = Zufallskomponente
#
# In diskreter Form:
# S(t+1) = S(t) * exp((μ - σ²/2) + σ * Z)
# Z ~ N(0,1) = Zufallszahl aus Normalverteilung
#
# ================================================================================


def analyze_monte_carlo(symbol: str, history_period: str = "1y", forecast_days: int = 30, num_simulations: int = 1000) -> dict:
    """
    Führt eine Monte-Carlo-Simulation für Kursprognosen durch.
    
    Diese Methode ist besonders nützlich für:
    - Risikobewertung ("Was könnte im schlimmsten Fall passieren?")
    - Wahrscheinlichkeitsanalyse ("Wie wahrscheinlich ist Gewinn?")
    - Szenario-Planung ("Was sind realistische Kursziele?")
    
    Args:
        symbol: Aktiensymbol (z.B. "TSLA")
        history_period: Zeitraum für Volatilitäts-Berechnung
        forecast_days: Anzahl Tage für die Simulation
        num_simulations: Anzahl der Simulationspfade (mehr = genauer, aber langsamer)
    
    Returns:
        dict: Chart, Statistiken, Wahrscheinlichkeiten, Perzentile
    """
    symbol = symbol.strip().upper()
    
    try:
        # === SCHRITT 1: Kursdaten abrufen ===
        stock = yf.Ticker(symbol)
        hist = stock.history(period=history_period)
        
        if hist.empty or len(hist) < 30:
            return {"error": f"Nicht genügend Kursdaten für '{symbol}'. Mindestens 30 Datenpunkte benötigt."}
        
        # === SCHRITT 2: Daten vorbereiten ===
        close_prices = hist["Close"].values
        dates_original = hist.index.tolist()
        
        # Zeitzonen entfernen
        dates_clean = []
        for d in dates_original:
            if hasattr(d, 'to_pydatetime'):
                dt = d.to_pydatetime()
            else:
                dt = pd.Timestamp(d).to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            dates_clean.append(dt)
        
        df = pd.DataFrame({
            "Close": close_prices,
            "Date": dates_clean
        })
        df = df.dropna()
        
        if len(df) < 30:
            return {"error": f"Nach Bereinigung nicht genügend Daten für '{symbol}'."}
        
        # === SCHRITT 3: Parameter für GBM berechnen ===
        # Log-Renditen: ln(P_t / P_{t-1})
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        
        # Drift (μ): Durchschnittliche tägliche Rendite
        mu = returns.mean()
        
        # Volatilität (σ): Standardabweichung der Renditen
        sigma = returns.std()
        
        # Aktueller Preis (Startpunkt der Simulation)
        current_price = df["Close"].iloc[-1]
        
        # === SCHRITT 4: Monte-Carlo Simulation durchführen ===
        dt = 1  # Zeitschritt = 1 Tag
        
        # Zufallsgenerator initialisieren (für reproduzierbare Ergebnisse)
        np.random.seed(42)
        
        # Array für alle Simulationspfade erstellen
        # Shape: (Anzahl Simulationen, Anzahl Tage + 1)
        simulations = np.zeros((num_simulations, forecast_days + 1))
        
        # Alle Simulationen starten beim aktuellen Kurs
        simulations[:, 0] = current_price
        
        # Für jeden Tag...
        for t in range(1, forecast_days + 1):
            # Zufallszahlen aus Normalverteilung N(0,1) ziehen
            # Eine Zufallszahl pro Simulation
            random_returns = np.random.normal(0, 1, num_simulations)
            
            # GBM Formel anwenden:
            # S(t) = S(t-1) * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)
            #
            # - (μ - 0.5*σ²): "Drift-Korrektur" - verhindert systematische Überschätzung
            # - σ*√dt*Z: Zufallskomponente, skaliert mit Volatilität
            simulations[:, t] = simulations[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_returns
            )
        
        # === SCHRITT 5: Statistiken aus den Simulationen berechnen ===
        # Endpreise aller Simulationen (letzter Tag)
        final_prices = simulations[:, -1]
        
        # Perzentile: "X% der Simulationen enden unter diesem Preis"
        percentiles = {
            "p5": np.percentile(final_prices, 5),    # 5% = Worst Case (fast)
            "p10": np.percentile(final_prices, 10),
            "p25": np.percentile(final_prices, 25),   # Unteres Quartil
            "p50": np.percentile(final_prices, 50),   # Median = "typisches" Ergebnis
            "p75": np.percentile(final_prices, 75),   # Oberes Quartil
            "p90": np.percentile(final_prices, 90),
            "p95": np.percentile(final_prices, 95),   # 95% = Best Case (fast)
        }
        
        # Durchschnitt und Standardabweichung
        mean_price = np.mean(final_prices)
        std_price = np.std(final_prices)
        
        # Wahrscheinlichkeiten berechnen
        # "In wie vielen Simulationen ist der Kurs gestiegen?"
        prob_positive = np.sum(final_prices > current_price) / num_simulations * 100
        
        # "In wie vielen Simulationen ist der Kurs >10% gestiegen?"
        prob_up_10 = np.sum(final_prices > current_price * 1.10) / num_simulations * 100
        
        # "In wie vielen Simulationen ist der Kurs >10% gefallen?"
        prob_down_10 = np.sum(final_prices < current_price * 0.90) / num_simulations * 100
        
        # === SCHRITT 6: Prognose-Daten erstellen ===
        # Nur Werktage für die X-Achse
        last_date = dates_clean[-1]
        forecast_date_list = []
        current_date = last_date + timedelta(days=1)
        while len(forecast_date_list) <= forecast_days:
            if current_date.weekday() < 5:  # 0-4 = Mo-Fr
                forecast_date_list.append(current_date)
            current_date = current_date + timedelta(days=1)
        
        # === SCHRITT 7: Chart erstellen ===
        fig = create_monte_carlo_chart(
            symbol, df, forecast_date_list, simulations, 
            percentiles, mean_price, current_price
        )
        
        # Erwartete Änderung (basierend auf Durchschnitt)
        forecast_change = ((mean_price - current_price) / current_price) * 100
        
        # === SCHRITT 8: Ergebnis zurückgeben ===
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
    """
    Gibt ein Label und eine Farbe für die Monte-Carlo-Wahrscheinlichkeit zurück.
    
    Interpretiert die Gewinnwahrscheinlichkeit für den Benutzer.
    Verwendet Börsen-Jargon (Bullisch/Bearisch):
    - Bullisch (Stier) = optimistisch, steigende Kurse erwartet
    - Bearisch (Bär) = pessimistisch, fallende Kurse erwartet
    
    Args:
        prob_positive: Wahrscheinlichkeit für Gewinn in Prozent (0-100)
    
    Returns:
        tuple: (Label-Text, Bootstrap-Farbe)
    
    Schwellenwerte:
    - > 70%: Sehr bullisch (starkes Kaufsignal)
    - 55-70%: Bullisch (leichtes Kaufsignal)
    - 45-55%: Neutral (abwarten)
    - 30-45%: Bearisch (leichtes Verkaufssignal)
    - < 30%: Sehr bearisch (starkes Verkaufssignal)
    """
    if prob_positive > 70:
        return "Sehr bullisch", "success"   # Grün
    elif prob_positive > 55:
        return "Bullisch", "success"
    elif prob_positive < 30:
        return "Sehr bearisch", "danger"    # Rot
    elif prob_positive < 45:
        return "Bearisch", "danger"
    return "Neutral", "secondary"           # Grau

# ================================================================================
# ENDE DES MODULS
# ================================================================================
# Dieses Modul wird von app_dash mit Kontostand.py importiert.
# Die Funktionen werden über Callbacks aufgerufen, wenn der Benutzer
# im AI-Analyse Tab auf "Analysieren" klickt.
#
# Zusammenfassung der exportierten Funktionen:
# - analyze_sentiment(): Sentiment-Analyse von News
# - analyze_correlation(): Korrelation Sentiment <-> Kurs
# - analyze_forecast(): ARIMA-basierte Kursprognose
# - analyze_monte_carlo(): Monte-Carlo-Simulation
# - get_sentiment_label(): Label für Sentiment-Score
# - get_correlation_label(): Label für Korrelation
# - get_forecast_label(): Label für Prognose
# - get_monte_carlo_label(): Label für MC-Ergebnis
#
# Verwendete Konstanten (auch exportiert):
# - VADER_AVAILABLE: Ist VADER installiert?
# - ARIMA_AVAILABLE: Ist statsmodels installiert?
# ================================================================================
