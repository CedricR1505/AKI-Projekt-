# 📈 Stock Dashboard - Interaktive Aktienanalyse- und Portfolio-Verwaltungsplattform

Eine vollständige Python-basierte Web-Anwendung zur **Echtzeit-Aktienanalyse**, **Portfolio-Management** und **KI-gestützte Prognosen** mit modernem interaktivem Dashboard.

## ✨ Features

### 📊 Portfolio-Management
- **Virtuelles Portfolio**: Starten Sie mit $10,000 virtuellem Kapital
- **Buy/Sell Funktionalität**: Kaufen und verkaufen Sie Aktien in Echtzeit
- **Portfolio-Tracking**: Überwachen Sie Ihre Positionen mit Live-Kursen
- **Gewinn/Verlust-Berechnung**: Sehen Sie Ihre Performance auf einen Blick
- **Transaktionshistorie**: Vollständige Dokumentation aller Trades
- **Portfolio-Charts**: Kreisdiagramme und Wertentwicklungs-Charts

### 📈 Aktienanalyse
- **Live Kursdaten**: Integration mit Yahoo Finance für Echtzeit-Aktiendaten
- **Interaktive Charts**: Zoombar, verschiebbar, mit Hover-Informationen
- **Mehrere Zeiträume**: 1T, 1W, 1M, 3M, 6M, 1J, Max
- **Markt-Übersicht**: DAX, MDAX, SDAX, Dow Jones, Nasdaq, Gold, Öl, BTC, EUR/USD
- **Aktiensearch**: Suche nach Aktien nach Symbol oder Firmenname

### 📰 Finanznachrichten
- **RSS-Feed Integration**: News aus Google News, Yahoo Finance, MarketWatch, CNBC, uvm.
- **Kategorien**: Alle, Aktien, Kryptowährungen, Wirtschaft
- **Nachrichten-Suche**: Filtern nach Aktien und Suchbegriffen
- **Kachel-Layout**: Moderne Darstellung mit Bildern und Hover-Effekten

### 🤖 AI-Analysen (Advanced Features)
#### 1. **Sentiment-Analyse**
- Analysiere die Stimmung zu Aktien basierend auf Nachrichten
- VADER Sentiment Analyzer für automatische Stimmungserkennung
- Vergleiche Sentiment mit Kursbewegungen
- Finde Korrelationen zwischen News-Stimmung und Kursänderungen

#### 2. **ARIMA-Kursprognose**
- Zeitreihen-basierte Vorhersage mit AutoRegressive Integrated Moving Average
- Automatische Parameteranpassung (p, d, q)
- 95% Konfidenzintervalle für Prognosen
- Trend-Korrektur für langfristige Vorhersagen
- Historische und prognostizierte Daten im selben Chart

#### 3. **Monte-Carlo-Simulation**
- Stochastische Kursprognose mit Geometric Brownian Motion
- Tausende Simulationspfade für Wahrscheinlichkeitsanalysen
- Risikoberechnung und Gewinn/Verlust-Wahrscheinlichkeiten
- Perzentile und Konfidenzintervalle (50%, 90%)

### 🎨 Benutzeroberfläche
- **Dark/Light Mode**: Wechsel zwischen dunklem und hellem Design
- **Responsive Design**: Funktioniert auf Desktop, Tablet und Smartphone
- **Bootstrap 5 Styling**: Modernes, professionelles Aussehen
- **Echtzeit-Updates**: Automatische Aktualisierung alle 15 Sekunden
- **Modals & Popovers**: Intuitive Dialoge für Transaktionen und Details

## 🚀 Installation & Setup

### Voraussetzungen
- Python 3.8+
- pip (Python Package Manager)

### Schritt 1: Repository klonen
```bash
git clone <your-repo-url>
cd AKI-Projekt-
```

### Schritt 2: Virtual Environment erstellen
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# oder
venv\Scripts\activate  # Windows
```

### Schritt 3: Dependencies installieren
```bash
pip install -r requirements.txt
```

### Schritt 4: Anwendung starten
```bash
python "app_dash mit Kontostand.py"
```

Die Anwendung läuft dann unter: **http://localhost:8050**

## 📦 Abhängigkeiten

```
dash==3.3.0                          # Web-Framework
dash-bootstrap-components==2.0.4    # Bootstrap UI Components
plotly==6.5.0                        # Interaktive Charts
yfinance==0.2.66                    # Yahoo Finance API
requests==2.32.5                    # HTTP Requests
pandas==2.3.3                        # Datenanalyse
feedparser==6.0.11                  # RSS Feed Parsing
vaderSentiment==3.3.2                # Sentiment Analysis
statsmodels>=0.14.0                  # ARIMA Modelle
numpy>=1.24.0                        # Numerische Berechnungen
```

## 📁 Projektstruktur

```
AKI-Projekt-/
├── app_dash mit Kontostand.py      # Hauptanwendung (Dash Frontend + Callbacks)
├── sentiment_analysis.py            # KI-Analyse Modul (2145 Zeilen)
├── test_arima.py                    # Test-Script für ARIMA-Funktionalität
├── requirements.txt                 # Python Dependencies
├── README.md                         # Diese Datei
├── assets/                           # Statische Assets
│   └── logo.png                     # Dashboard Logo
└── gui/                             # Datenspeicher (JSON Files)
    ├── portfolio.json               # Portfolio-Positionen
    ├── transactions.json            # Transaktionshistorie
    └── balance.json                 # Kontostand
```

## 🎯 Hauptdateien erklärt

### `app_dash mit Kontostand.py` (3840 Zeilen)
Die **Hauptanwendung** mit:
- Dash/Flask Web-Server-Initialisierung
- Komplett HTML-Layout mit Tabs und Komponenten
- Alle Callbacks für Benutzerinteraktionen
- Portfolio-Management Funktionen (Buy/Sell)
- Chart-Erstellung mit Plotly
- Daten-Persistenz (JSON-basiert)

**Wichtige Funktionen:**
- `fetch_price(symbol)` - Aktuelle Kurse abrufen
- `fetch_stock_history(symbol, period, interval)` - Historische Daten
- `search_stocks(query)` - Aktiensuche
- `fetch_google_news(symbol)` - News-Abruf
- `create_stock_chart(symbol, period)` - Chart-Erstellung
- `create_portfolio_pie_chart(portfolio)` - Portfolio-Visualisierung

### `sentiment_analysis.py` (2145 Zeilen)
Das **KI-Analyse Modul** mit:
- News-Abruf aus 15+ RSS-Feed-Quellen
- VADER-basierte Sentiment-Analyse
- Korrelationsberechnung (Sentiment vs. Kurs)
- ARIMA-Zeitreihen-Prognose
- Monte-Carlo-Simulation
- Chart-Generierung für Analysen

**Wichtige Funktionen:**
- `fetch_news_from_feeds(symbol, period, news_limit)` - News sammeln
- `analyze_sentiment(symbol, period)` - Stimmungsanalyse
- `analyze_correlation(symbol, period)` - Korrelationsanalyse
- `analyze_forecast(symbol, history_period, forecast_days)` - ARIMA Prognose
- `analyze_monte_carlo(symbol, history_period, forecast_days, num_simulations)` - MC Simulation

### `test_arima.py`
Test-Script zur Validierung der ARIMA-Funktionalität mit Apple (AAPL) Daten.

## 💰 Verwendung

### 1. Portfolio Management
1. Gehe zu Tab **"Portfolio"**
2. Klicke auf **"💰 Buy/Sell"**
3. Gebe Symbol (z.B. AAPL) und Anzahl Aktien ein
4. Wähle "Buy" oder "Sell"
5. Dein Portfolio wird aktualisiert

### 2. Aktienanalyse
1. Gehe zu Tab **"Aktien"**
2. Suche nach einer Aktie (z.B. "Apple", "TSLA")
3. Wähle einen Zeitraum (1T, 1W, 1M, etc.)
4. Sehe Live-Chart und News

### 3. AI-Analysen
1. Gehe zu Tab **"AI Analysis"**
2. Wähle eine Analyse:
   - **Sentiment-Analyse**: Stimmung von News analysieren
   - **ARIMA**: Statistisch basierte Kursprognose
   - **Monte-Carlo**: Wahrscheinlichkeitsanalyse

## 🔍 Technische Details

### Sentiment-Analyse Workflow
1. RSS-Feeds abrufen (Google News, Yahoo Finance, etc.)
2. News für Symbol filtern
3. VADER Sentiment Score für jeden Titel berechnen (-1 bis +1)
4. Tägliche Durchschnitte bilden
5. Kursdaten mit Yahoo Finance abrufen
6. Korelation berechnen (Pearson-Korrelationskoeffizient)
7. Visualisierung mit Plotly

### ARIMA Prognose
1. Historische Kursdaten laden
2. Log-Renditen berechnen
3. Stationarität mit ADF-Test prüfen
4. Parameter-Grid-Search (p, d, q) durchführen
5. Best AIC Model wählen
6. Forecast mit Trend-Korrektur erstellen
7. 95% Konfidenzintervalle berechnen

### Monte-Carlo Simulation
1. Historische Log-Renditen berechnen
2. Drift (μ) und Volatilität (σ) berechnen
3. Geometric Brownian Motion für jeden Pfad:
   - S(t+1) = S(t) * exp((μ - σ²/2) + σ*Z)
4. Tausende Simulationen durchführen
5. Perzentile und Wahrscheinlichkeiten berechnen

## 🎨 Design Features

- **Dark/Light Mode**: Benutzerwahl zwischen Themes
- **Responsive Grid**: Passt sich Bildschirmgröße an
- **Color Coding**: Grün für Gewinne, Rot für Verluste
- **Hover Effects**: Interaktive Visualisierungen
- **Bootstrap Components**: Modernes UI mit Buttons, Modals, Alerts

## ⚠️ Wichtige Hinweise

### Disclaimer
- **Kein echtes Geld**: Alle Transaktionen sind virtuell
- **Keine Empfehlungen**: Dashboard ist zu Lernzwecken
- **Prognosen ungenau**: Finanzprognosen sind immer fehlerhaft
- **APIs können ausfallen**: Yahoo Finance und RSS-Feeds können begrenzt sein

### Performance
- Sentiment-Analyse kann bei vielen News langsam sein
- Monte-Carlo mit 10000+ Simulationen braucht Zeit
- Große Charts können beim Zoomen laggen

### Datenspeicherung
- Portfolio wird lokal in `gui/portfolio.json` gespeichert
- Keine Cloud-Synchronisation
- Daten gehen bei Löschen des `gui/`-Ordners verloren

## 🛠️ Troubleshooting

### "Module 'vaderSentiment' not found"
```bash
pip install vaderSentiment
```

### "No data for symbol X"
- Symbol ist ungültig oder Yahoo Finance kennt ihn nicht
- Versuchen Sie ein anderes Symbol

### Charts laden nicht
- Netzwerkfehler bei Yahoo Finance
- Versuchen Sie F5 zum Aktualisieren

### ARIMA Fehler
- Zu wenig historische Daten (min. 30 Datenpunkte)
- Symbol existiert nicht

## 🚀 Zukünftige Erweiterungen

- [ ] Echte Broker-Integration (z.B. Alpaca)
- [ ] Benutzer-Authentifizierung & Cloud-Speicher
- [ ] Erweiterte technische Indikatoren (RSI, MACD, Bollinger Bands)
- [ ] Options-Analysen
- [ ] Backtesting für Handelsstrategien
- [ ] Webhook-Benachrichtigungen
- [ ] Export zu PDF/Excel

## 📖 Ressourcen

- [Dash Dokumentation](https://dash.plotly.com/)
- [Plotly Charts](https://plotly.com/python/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [yfinance](https://github.com/ranaroussi/yfinance)

## 👨‍💻 Autor

Entwickelt als **AKI-Projekt** (Anwendung von künstlicher Intelligenz in der Finanztechnik)

## 📄 Lizenz

MIT License 

---

**Hinweis**: Dies ist ein Bildungsprojekt. Verwenden Sie es nicht für echte Investitionsentscheidungen ohne professionelle Beratung!
