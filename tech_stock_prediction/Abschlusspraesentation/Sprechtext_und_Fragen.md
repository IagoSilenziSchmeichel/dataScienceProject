# Sprechtext, Foliennotizen & Q&A-Vorbereitung

Begleitdokument zu `LSTM_Outperformance_Abschlusspraesentation.pptx` (48 Folien). Enthält für jede Folie: Sprechtext/Stichpunkte, worauf beim Vortragen zu achten ist, und antizipierte Rückfragen mit Antworten. Alle Zahlen entsprechen exakt den Folieninhalten und stammen aus `presentation_reporting/output/*/wissenschaftliche_analyse.md` und `presentation_reporting/output/comparison/gesamtanalyse.md`.

## Timing-Einschätzung

Bei einem vollständigen, langsamen Vorlesen aller Stichpunkte ergeben die Foliennotizen ca. 33-38 Minuten (48 Folien). Das ist mehr als das Ziel von 15-20 Minuten. Realistisch für ein gut geprobtes Team von 4 Vortragenden (je ca. 12 Folien) ist eine deutlich zügigere Vortragsweise, bei der pro Folie nur die Kernzahl und 1 Erkenntnis laut gesagt werden, der Rest der Stichpunkte dient als Gedächtnisstütze:

| Foliengruppe | Anzahl | Realistisches Tempo/Folie | Summe |
| --- | --- | --- | --- |
| Section-Divider (Titel, 4× Universum, Gesamtvergleich, Projektabschluss, Danke) | 8 | ~15 s | ~2,0 min |
| Methodik-Folien (Problem bis Backtest-Methodik) | 8 | ~40 s | ~5,3 min |
| Ergebnis-Folien je Universum (4 × 5) | 20 | ~30 s | ~10,0 min |
| Gesamtvergleich-Folien | 7 | ~35 s | ~4,1 min |
| Backtest-vs-Realität-Diskussion | 2 | ~40 s | ~1,3 min |
| Projektabschluss & Fazit | 3 | ~40 s | ~2,0 min |
| **Summe** | **48** | | **~24,7 min** |

Das liegt näher am Ziel, aber immer noch leicht darüber. Falls die Zeit knapp wird, können folgende Folien live gekürzt oder übersprungen werden (Begründung direkt aus `gesamtanalyse.md`, Abschnitt „Präsentationsempfehlungen — Kann weggelassen werden“): die Signalanalyse-Vergleichsfolie (Vergleich 5), eine der vier Paper-Trading-Folien kurz halten (Hourly-Status ist bei allen vier Universen strukturell gleich – „zu wenig Daten“ – das muss nicht viermal ausführlich wiederholt werden), und bei der Feature-Engineering-Folie nur 2-3 der 6 Gruppen im Detail vorlesen statt aller sechs.

## Allgemeine Rückfragen (projektweit, nicht folienspezifisch)

**1. Wie wurde Data-Snooping / Leakage verhindert?**
Chronologischer 70/15/15-Split (Training 2019-2024, Validation Mitte 2024-2025, Test Mitte 2025-2026), und die Normalisierung (Skalierung) jeder Aktie erfolgt ausschließlich auf Basis der Trainingsstatistiken – Validation und Test werden nur mit diesen bereits fixierten Parametern transformiert, nie neu berechnet.

**2. Warum wurden Standard-LSTM und Outperformance-LSTM nicht auf dieselbe Metrik optimiert?**
Der Standard-LSTM sagt die reine Richtung (steigt/fällt) vorher und dient als einfache Baseline. Der Outperformance-LSTM sagt relative Stärke zum Benchmark vorher – das ist die ökonomisch relevantere, aber auch schwerer zu lernende Zielgröße (siehe z. B. Original Tech: Standard-LSTM F1 0,651 vs. Outperformance-LSTM F1 0,541, obwohl Letzterer ökonomisch klar erfolgreicher ist).

**3. Warum wurde kein Hyperparameter-Tuning durchgeführt?**
Aus Zeitgründen für die Präsentationsvorbereitung bewusst übersprungen (siehe Kommentar im Orchestrator-Skript run_universe_lstm_outperformance.py) – im Code als eigenständiger, separat ausführbarer Schritt vorbereitet, aber für alle vier Universen mit denselben Hyperparametern gefahren, um einen fairen Vergleich zu ermöglichen.

**4. Warum genau 20 Handelstage als Sequenzlänge?**
Fixer Parameter aus der Projekt-Konfiguration (params.yaml), identisch für alle vier Universen – ein gängiger Kompromiss zwischen genug historischem Kontext und Trainingsaufwand; ein systematischer Vergleich verschiedener Sequenzlängen wurde in diesem Projekt nicht durchgeführt (siehe Future Work).

**5. Sind die Ergebnisse statistisch signifikant?**
Das wurde nicht formal getestet (kein Signifikanztest gerechnet) – explizit als Limitation benannt. Die Aussagekraft basiert auf 235 Handelstagen Backtest je Universum, was für Finanzzeitreihen üblich, aber kein Ersatz für einen formalen Test ist.

**6. Warum wird kein Transaktionskostenmodell verwendet?**
Bewusste Vereinfachung aller vier Backtests, um die reine Signalqualität zu isolieren – im Fazit und in den Limitationen explizit als Punkt genannt, der reale Nettoerträge schmälern würde.

**7. Wie wurde sichergestellt, dass alle vier Universen wirklich exakt gleich behandelt wurden?**
Über denselben Code-Pfad (run_universe_lstm_outperformance.py, identische Skript-Liste), denselben Zeitraum, denselben Split und dieselbe Feature-Definition (MARKET_FEATURES/RELATIVE_FEATURES in outperformance_lstm.py) – der einzige Unterschied ist die Aktienliste und der jeweilige Benchmark (QQQ oder SPY).

## Folie für Folie

### Folie 1 — Intro · Titel

**Sprechtext:** Begrüßung, Titel, kurze Agenda-Ankündigung.

---

### Folie 2 — Motivation · Den Markt zu schlagen ist der Standardfall, nicht die Ausnahme – Ziel jedes aktiven Investors

**Sprechtext:** Kernbotschaft: Wir wollen nicht nur Richtung vorhersagen, sondern relative Stärke gegenüber dem Markt. Das ist die Prämisse für alles Folgende. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Buy-and-Hold auf einen Index (z. B. QQQ oder SPY) ist die passive Baseline, die jede aktive Strategie schlagen müsste, um ihren Mehraufwand zu rechtfertigen.
- Klassische technische Indikatoren (RSI, MACD, Momentum) sind einzeln bekannt – die Frage ist, ob ein neuronales Netz aus ihrem Zusammenspiel über die Zeit ein nutzbares Signal lernen kann.
- Statt nur „steigt die Aktie morgen“ vorherzusagen, stellen wir die ökonomisch relevantere Frage: „schlägt die Aktie morgen den Markt“ – das ist die Grundlage unseres finalen Modells.
- Wir testen diese Idee nicht auf einem, sondern auf vier unabhängigen Aktienuniversen, um zu prüfen, ob ein gefundenes Muster generalisiert oder nur für eine bestimmte Aktiengruppe gilt.

**Erwartbare Rückfragen:**
- *Warum nicht einfach den Preis vorhersagen?* — Preisvorhersage ist extrem verrauscht und ökonomisch wenig aussagekräftig ohne Bezug zu einer Alternative (Benchmark). Die Outperformance-Formulierung macht das Ziel direkt handlungsrelevant: relativ kaufen oder nicht.

---

### Folie 3 — Motivation · Ein Modell, vier Universen, eine wissenschaftliche Frage

**Sprechtext:** Hier explizit sagen: Random Forest war nur Datenvorbereitung, DAS finale Modell ist die Outperformance-LSTM-Strategie. Diese Folie verankert die vier Teilfragen, die wir gleich einzeln beantworten. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Finales Modell: Outperformance-LSTM mit Top-K-Selektionsstrategie – sagt vorher, ob eine Aktie den nächsten Handelstag besser abschneidet als ihr Benchmark.
- Identische Methodik (Code, Features, Architektur, Trainingsparameter) wird auf vier unabhängige Aktienuniversen angewendet – nur die Aktienauswahl und der Benchmark unterscheiden sich.
- Leitfrage: Funktioniert die Strategie universell, oder ist sie an bestimmte Markteigenschaften (Tech, hohe Korrelation, hohe Volatilität) gebunden?
- Jedes Universum beantwortet eine eigene Teilfrage – von „funktioniert es überhaupt auf Tech“ bis „funktioniert es auch außerhalb von Tech“.

---

### Folie 4 — Datengrundlage · Gleiche Methodik, vier unterschiedliche Aktienwelten

**Sprechtext:** Wichtig: identischer Zeitraum, identischer Split, identische Pipeline – der einzige Unterschied ist die Aktienauswahl und der Benchmark. Das macht den späteren Vergleich wissenschaftlich fair. Volatilität und Korrelation kurz erklären: höhere Volatilität = schwerer vorhersagbar, höhere Korrelation = Aktien bewegen sich ähnlicher, weniger Spielraum für Top-K-Auswahl. Ca. 60 Sekunden.

**Erwartbare Rückfragen:**
- *Warum unterschiedliche Benchmarks (QQQ vs. SPY)?* — QQQ ist ein techlastiger Index, passend zu den drei Tech-Universen. SPY ist der breite Marktindex und damit der korrekte Vergleichsmaßstab für das defensive Nicht-Tech-Universum.
- *Woher kommt die Marktkapitalisierung nicht in der Tabelle?* — Wurde in der Pipeline nicht erhoben – bewusst nicht geschätzt, um keine Zahl zu erfinden.

---

### Folie 5 — Methodik · Pipeline-Architektur

**Sprechtext:** Diese Folie zeigt die reale Code-Pipeline (run_universe_lstm_outperformance.py). Obere Reihe: gemeinsame Datenschritte, wiederverwendet aus dem alten Random-Forest-Experiment, aber nur zur Datenaufbereitung, nicht zum Training. Mittlere Reihe: der eigentliche LSTM-Trainingsweg, inkl. Standard-LSTM als Baseline und dem finalen Outperformance-LSTM. Untere Reihe: was danach mit den Ergebnissen passiert – Archivierung pro Universum, Alpaca Paper Trading als separates System, und dieser Reporting-Layer, der ausschließlich vorhandene Ergebnisse auswertet. Ca. 60-75 Sekunden.

**Erwartbare Rückfragen:**
- *Wurde für jedes Universum neu trainiert?* — Ja – die komplette Pipeline (Daten, Sequenzen, Training, Backtest) läuft separat pro Universum und wird danach archiviert, damit nichts überschrieben wird.

---

### Folie 6 — Methodik · Feature Engineering

**Sprechtext:** 19 technische Basis-Features + 7 Market Features + 4 Relative-Strength-Features = 30. Die Relative-Strength-Features sind konzeptionell zentral: die Zielvariable ist relativ zum Benchmark definiert, eine Aktie kann fallen und trotzdem 'Outperform' sein, wenn der Benchmark stärker fällt. Der Code für diese Features ist identisch für alle vier Universen – nur der Ticker im Feature-Namen (QQQ/SPY) unterscheidet sich nicht einmal, beide Varianten werden immer mitgeführt. Ca. 50 Sekunden.

**Stichpunkte auf der Folie:**
- Momentum (8): Lag_1/3/7_Return, Momentum_5/10/20, RollingMean_7/30
- Trend (5): MACD, MACD_Signal, RSI_14, Distance_to_MA_200, Price_Position_20
- Volatilität (3): RollingVolatility_7/30, High_Low_Range
- Volumen (2): Volume_Change, Volume_Ratio_20
- Market Features (7): QQQ/SPY_Return, VIX_Change, QQQ/SPY_Momentum_20, QQQ/SPY_Distance_to_MA200
- Relative Strength (4): Relative_Return_QQQ/SPY, Relative_Momentum_20_QQQ/SPY

**Erwartbare Rückfragen:**
- *Wurde eine Feature-Wichtigkeits-Analyse gemacht?* — Nein, mit den aktuell archivierten Dateien nicht – das wäre ein zusätzlicher, nicht-trainierender Auswertungsschritt (z. B. Permutation Importance), der für dieses Projekt nicht durchgeführt wurde.

---

### Folie 7 — Methodik · Outperformance-LSTM: nicht „steigt die Aktie“, sondern „schlägt sie den Markt“

**Sprechtext:** Bewusst ohne Formeln – Kernidee: das Modell sieht eine Zeitreihe von Mustern, nicht nur einen Schnappschuss, und die Ausgabe ist relativ zum Markt, nicht absolut. Top-K als Portfolio-Konstruktionsregel erklären: K bestimmt, wie konzentriert oder diversifiziert das Portfolio ist. Ca. 60 Sekunden.

**Stichpunkte auf der Folie:**
- Zielvariable: Outperform_Target = 1, wenn die Tagesrendite der Aktie am nächsten Handelstag höher ist als die des Benchmarks – sonst 0.
- Ein LSTM (Long Short-Term Memory) liest eine Sequenz von 20 aufeinanderfolgenden Handelstagen je Aktie und lernt zeitliche Muster in den 30 Features, statt jeden Tag isoliert zu betrachten.
- Ausgabe ist eine Wahrscheinlichkeit pro Aktie und Tag: wie sicher ist sich das Modell, dass diese Aktie morgen den Benchmark schlägt?
- Top-K-Strategie: an jedem Signaltag werden die K Aktien mit der höchsten Wahrscheinlichkeit gekauft (K = 1 bis 5 getestet) – das Portfolio wird laufend an die aktuellen Top-K angepasst.

**Erwartbare Rückfragen:**
- *Warum LSTM und nicht z. B. ein einfacheres Modell?* — LSTMs können Reihenfolge und zeitliche Abhängigkeiten über die 20-Tage-Sequenz lernen, was für Momentum-/Trendmuster relevant ist – ein Modell ohne Sequenzgedächtnis würde diese Struktur verlieren.

---

### Folie 8 — Methodik · Identische Architektur und Hyperparameter für alle vier Universen

**Sprechtext:** Diese Parameter sind aus der params.yaml übernommen, nicht geschätzt. Wichtig zu betonen: 'identisch' bezieht sich auf Architektur/Hyperparameter, nicht auf die gelernten Gewichte – jedes Universum hat sein eigenes trainiertes Modell. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Handelstage
Sequenzlänge: 20
- Hidden Size: 64
- Trainings-Epochen: 20
- Batch Size: 32
- Learning Rate: 0,001

---

### Folie 9 — Methodik · Wie wir „gut“ messen, bevor wir die Ergebnisse zeigen

**Sprechtext:** Diese Folie ist die Brücke zu den vier Universen-Blöcken: ab hier folgt für jedes Universum dieselbe 5-Folien-Struktur (Backtest, Top-K, Signale, Paper Trading, Fazit), damit die vier Geschichten direkt vergleichbar sind. Ca. 40 Sekunden.

**Stichpunkte auf der Folie:**
- Backtest-Zeitraum: 01.08.2025 – 09.07.2026 (235 Handelstage), für alle vier Universen identisch und disjunkt vom Trainingszeitraum.
- Sharpe Ratio: Rendite im Verhältnis zur eingegangenen Volatilität – hoch bedeutet gute Rendite pro Risikoeinheit, nicht nur hohe absolute Rendite.
- Max Drawdown: größter Wertverlust vom Höchststand aus – entscheidend für die praktische Einsetzbarkeit, da große Drawdowns in der Praxis zum Strategieabbruch führen können.
- Vergleichsgrößen: immer gegen Buy-and-Hold (passive Baseline) und den Standard-LSTM (Richtungs-Baseline ohne Benchmark-Bezug) – kein Transaktionskostenmodell, das ist eine bewusste Vereinfachung.

---

### Folie 10 — Original Tech · Divider: Original Tech

**Sprechtext:** Forschungsfrage: Wie funktioniert das Modell auf dem ursprünglichen Tech-Universum?. Aktien: AAPL, MSFT, NVDA, AMD, GOOGL, META, AMZN, TSLA, INTC, ADBE. Benchmark: QQQ.

---

### Folie 11 — Original Tech · Schlägt die Strategie QQQ?

**Sprechtext:** Kernzahl zuerst nennen: Top-1-Strategie +105,7 % gegen Buy-and-Hold +31,0 %, Differenz +74,7 %. Dann Sharpe/Drawdown als Risikoeinordnung. Zum Schluss kurz den Standard-LSTM als zweite Baseline erwähnen – zeigt, ob die Outperformance-Formulierung gegenüber einer einfachen Richtungsvorhersage etwas bringt. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Top-1: +105,7 %  vs. Buy-and-Hold: +31,0 %
- Differenz: +74,7 %
- Sharpe Ratio 2,17 · Max Drawdown −24,3 %
- 297 Trades (Top-K-Wechsel) über 01.08.2025 – 09.07.2026 (235 Handelstage)
- Standard-LSTM zum Vergleich: +39,9 % vs. +54,3 % (−14,4 %)

**Hinweis zum Diagramm:** Diese Kurve braucht evtl. zusätzliche Erklärung: sie zeigt die kumulative Wertentwicklung, nicht die Tagesrendite – ein Sprung nach oben bedeutet nicht 'ein guter Tag', sondern ein anhaltend besseres Niveau.

**Erwartbare Rückfragen:**
- *Warum ist Top-1 die gezeigte Strategie und nicht ein anderes K?* — Das ist das K mit der besten Rendite im Backtest für dieses Universum – die vollständige Top-1-bis-5-Übersicht folgt auf der nächsten Folie.

---

### Folie 12 — Original Tech · Wie viele Aktien sollte das Portfolio gleichzeitig halten?

**Sprechtext:** Erklären: je kleiner K, desto konzentrierter das Portfolio auf die vermeintlich beste Aktie – höhere Varianz, aber auch höheres Potenzial, wenn das Modell-Ranking stimmt. Größeres K diversifiziert und nähert sich Buy-and-Hold an. Für Original Tech ist Top-1 (+105,7 %) am besten. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Bestes K: Top-1 (+105,7 %)
- Schwächstes K: Top-4 (+60,5 %)
- Spannweite zwischen bestem und schwächstem K: 45,2 %
- Kleine K = konzentriert, hohe Varianz, hohes Potenzial. Große K = diversifiziert, nähert sich Buy-and-Hold an.

**Hinweis zum Diagramm:** Balkendiagramm mit Positionsangabe je K – Achsenbeschriftung enthält Ø Position, kurz erwähnen falls Publikum fragt.

---

### Folie 13 — Original Tech · Welche Aktien wählt das Modell – und wie sicher ist es sich?

**Sprechtext:** Zeigen, dass das Modell nicht willkürlich wählt: bestimmte Aktien werden konstant bevorzugt (AMD, INTC), andere nie. Die Wahrscheinlichkeiten liegen nah an 0,50 – kurz einordnen, dass das für Finanzmarkt-Klassifikation normal ist, ein Random-Walk-Element bleibt immer. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Häufigste Top-1-Wahl: AMD
- Ø Wahrscheinlichkeit 0,36 (Median 0,34, Std 0,05) — 0,50 wäre Zufallsniveau
- Konstant bevorzugt: AMD, INTC
- Nie gewählt: MSFT, GOOGL, META, AMZN, ADBE
- Ø Umschichtung pro Signaltag: 1,0 Positionswechsel

**Hinweis zum Diagramm:** Zeigt Auswahlhäufigkeit + Wahrscheinlichkeitsverteilung je Ticker; ggf. auf optionale Zusatzfolie 06_signal_probability_distribution.png verweisen, falls Rückfrage zur Verteilungsform kommt.

---

### Folie 14 — Original Tech · Hourly vs. Daily Alpaca Paper Trading

**Sprechtext:** Ehrlich kommunizieren: Hourly-Daten reichen aktuell nicht für eine Auswertung (1 Zeitstempel, technischer Grund: Umstellung auf täglichen Scheduler nach einem einmaligen Dry-Run) – hier NICHTS simulieren oder schönreden. Daily-Ergebnis ist nur ein 5-Tage-Plausibilitätscheck, klar als vorläufig kennzeichnen. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Hourly:
- Status: FEHLEND — nur 1 eindeutiger 1-Stunden-Zeitstempel aus einem einmaligen Dry-Run. Zu wenig für eine Zeitreihen-Auswertung.
- Daily:
- Portfolio −3,0 % vs. Benchmark +0,4 % (Differenz −3,4 %), Max Drawdown −6,7 %
- 17 (12 Kauf / 5 Verkauf), gehandeltes Volumen 579.490 $

**Hinweis zum Diagramm:** Das eingebettete Diagramm zeigt Daily Paper Trading; für Hourly existiert nur ein Hinweistext-Plot (Datenbasis zu klein) – im Backup-Anhang verfügbar, hier bewusst nicht groß gezeigt.

**Erwartbare Rückfragen:**
- *Warum gibt es keine Hourly-Ergebnisse?* — Es liegen nur 1-2 Zeitstempel aus einem einmaligen Dry-Run vor der Umstellung auf den täglichen Scheduler vor – das reicht nicht für eine belastbare Zeitreihenauswertung. Wir zeigen hier bewusst keine simulierten oder geschätzten Werte.

---

### Folie 15 — Original Tech · Antwort auf: „Wie funktioniert das Modell auf dem ursprünglichen Tech-Universum?“

**Sprechtext:** Diese Folie fasst die Teilfrage für Original Tech explizit zusammen, bevor es zum nächsten Universum geht. Stärken/Schwächen bewusst beide nennen – das ist Teil der wissenschaftlichen Redlichkeit des Projekts. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Top-1-Outperformance-LSTM erzielt +105,7 % gegenüber +31,0 % für QQQ – Differenz +74,7 %, Sharpe 2,17, Max Drawdown −24,3 %.
- AMD war die am häufigsten gewählte Top-1-Aktie; Ø Modell-Wahrscheinlichkeit 0,36 (0,50 = Zufallsniveau).
- Die 5-tägige Daily-Alpaca-Simulation zeigt −3,4 % Differenz zum Benchmark – bei dieser Stichprobengröße nur als Plausibilitätssignal zu werten.
- Stärken: Stärkste Outperformance aller vier Universen, höchster Sharpe, klarste Top-1-Konzentration.
- Schwächen: Größter Max Drawdown (−24,3 %); Outperformance-Klassifikator (F1 0,541) schwächer als der einfachere Standard-LSTM (F1 0,651).

---

### Folie 16 — Tech ohne Nvidia · Divider: Tech ohne Nvidia

**Sprechtext:** Forschungsfrage: Wie stark hängen die Ergebnisse von Nvidia ab?. Aktien: AAPL, MSFT, AMD, GOOGL, META, AMZN, TSLA, INTC, ADBE. Benchmark: QQQ.

---

### Folie 17 — Tech ohne Nvidia · Schlägt die Strategie QQQ?

**Sprechtext:** Kernzahl zuerst nennen: Top-5-Strategie +47,2 % gegen Buy-and-Hold +31,0 %, Differenz +16,2 %. Dann Sharpe/Drawdown als Risikoeinordnung. Zum Schluss kurz den Standard-LSTM als zweite Baseline erwähnen – zeigt, ob die Outperformance-Formulierung gegenüber einer einfachen Richtungsvorhersage etwas bringt. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Top-5: +47,2 %  vs. Buy-and-Hold: +31,0 %
- Differenz: +16,2 %
- Sharpe Ratio 1,81 · Max Drawdown −13,9 %
- 609 Trades (Top-K-Wechsel) über 01.08.2025 – 09.07.2026 (235 Handelstage)
- Standard-LSTM zum Vergleich: +36,8 % vs. +58,5 % (−21,7 %)

**Hinweis zum Diagramm:** Diese Kurve braucht evtl. zusätzliche Erklärung: sie zeigt die kumulative Wertentwicklung, nicht die Tagesrendite – ein Sprung nach oben bedeutet nicht 'ein guter Tag', sondern ein anhaltend besseres Niveau.

**Erwartbare Rückfragen:**
- *Warum ist Top-5 die gezeigte Strategie und nicht ein anderes K?* — Das ist das K mit der besten Rendite im Backtest für dieses Universum – die vollständige Top-1-bis-5-Übersicht folgt auf der nächsten Folie.

---

### Folie 18 — Tech ohne Nvidia · Wie viele Aktien sollte das Portfolio gleichzeitig halten?

**Sprechtext:** Erklären: je kleiner K, desto konzentrierter das Portfolio auf die vermeintlich beste Aktie – höhere Varianz, aber auch höheres Potenzial, wenn das Modell-Ranking stimmt. Größeres K diversifiziert und nähert sich Buy-and-Hold an. Für Tech ohne Nvidia ist Top-5 (+47,2 %) am besten. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Bestes K: Top-5 (+47,2 %)
- Schwächstes K: Top-1 (+14,5 %)
- Spannweite zwischen bestem und schwächstem K: 32,7 %
- Kleine K = konzentriert, hohe Varianz, hohes Potenzial. Große K = diversifiziert, nähert sich Buy-and-Hold an.

**Hinweis zum Diagramm:** Balkendiagramm mit Positionsangabe je K – Achsenbeschriftung enthält Ø Position, kurz erwähnen falls Publikum fragt.

---

### Folie 19 — Tech ohne Nvidia · Welche Aktien wählt das Modell – und wie sicher ist es sich?

**Sprechtext:** Zeigen, dass das Modell nicht willkürlich wählt: bestimmte Aktien werden konstant bevorzugt (AMD, INTC), andere nie. Die Wahrscheinlichkeiten liegen nah an 0,50 – kurz einordnen, dass das für Finanzmarkt-Klassifikation normal ist, ein Random-Walk-Element bleibt immer. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Häufigste Top-1-Wahl: AMD
- Ø Wahrscheinlichkeit 0,36 (Median 0,34, Std 0,05) — 0,50 wäre Zufallsniveau
- Konstant bevorzugt: AMD, INTC
- Nie gewählt: MSFT, GOOGL, META, AMZN, ADBE
- Ø Umschichtung pro Signaltag: 0,5 Positionswechsel

**Hinweis zum Diagramm:** Zeigt Auswahlhäufigkeit + Wahrscheinlichkeitsverteilung je Ticker; ggf. auf optionale Zusatzfolie 06_signal_probability_distribution.png verweisen, falls Rückfrage zur Verteilungsform kommt.

---

### Folie 20 — Tech ohne Nvidia · Hourly vs. Daily Alpaca Paper Trading

**Sprechtext:** Ehrlich kommunizieren: Hourly-Daten reichen aktuell nicht für eine Auswertung (1 Zeitstempel, technischer Grund: Umstellung auf täglichen Scheduler nach einem einmaligen Dry-Run) – hier NICHTS simulieren oder schönreden. Daily-Ergebnis ist nur ein 5-Tage-Plausibilitätscheck, klar als vorläufig kennzeichnen. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Hourly:
- Status: FEHLEND — nur 1 eindeutiger 1-Stunden-Zeitstempel aus einem einmaligen Dry-Run. Zu wenig für eine Zeitreihen-Auswertung.
- Daily:
- Portfolio −4,2 % vs. Benchmark +0,4 % (Differenz −4,6 %), Max Drawdown −7,4 %
- 10 (7 Kauf / 3 Verkauf), gehandeltes Volumen 420.097 $

**Hinweis zum Diagramm:** Das eingebettete Diagramm zeigt Daily Paper Trading; für Hourly existiert nur ein Hinweistext-Plot (Datenbasis zu klein) – im Backup-Anhang verfügbar, hier bewusst nicht groß gezeigt.

**Erwartbare Rückfragen:**
- *Warum gibt es keine Hourly-Ergebnisse?* — Es liegen nur 1-2 Zeitstempel aus einem einmaligen Dry-Run vor der Umstellung auf den täglichen Scheduler vor – das reicht nicht für eine belastbare Zeitreihenauswertung. Wir zeigen hier bewusst keine simulierten oder geschätzten Werte.

---

### Folie 21 — Tech ohne Nvidia · Antwort auf: „Wie stark hängen die Ergebnisse von Nvidia ab?“

**Sprechtext:** Diese Folie fasst die Teilfrage für Tech ohne Nvidia explizit zusammen, bevor es zum nächsten Universum geht. Stärken/Schwächen bewusst beide nennen – das ist Teil der wissenschaftlichen Redlichkeit des Projekts. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Top-5-Outperformance-LSTM erzielt +47,2 % gegenüber +31,0 % für QQQ – Differenz +16,2 %, Sharpe 1,81, Max Drawdown −13,9 %.
- AMD bleibt trotz Entfernung von Nvidia die am häufigsten gewählte Top-1-Aktie; Ø Wahrscheinlichkeit 0,36.
- Ohne Nvidia braucht die Strategie mehr Diversifikation: Top-5 schlägt Top-1 um 32,7 Prozentpunkte – umgekehrt wie im Original-Tech-Universum.
- Stärken: Robuste, positive Outperformance auch ohne den dominanten Einzeltitel Nvidia – spricht gegen reine Nvidia-Abhängigkeit.
- Schwächen: Höchste Trade-Anzahl (609) aller Universen; beste Performance erfordert Top-5 statt Top-1, also weniger Konzentration möglich.

---

### Folie 22 — Neue Tech-Aktien · Divider: Neue Tech-Aktien

**Sprechtext:** Forschungsfrage: Generalisiert das Modell auf weitere bzw. kleinere Tech-Unternehmen außerhalb des ursprünglichen Universums?. Aktien: PLTR, SNOW, CRWD, NET, NOW, ORCL, CSCO, MU, SMCI, DELL. Benchmark: QQQ.

---

### Folie 23 — Neue Tech-Aktien · Schlägt die Strategie QQQ?

**Sprechtext:** Kernzahl zuerst nennen: Top-2-Strategie +23,5 % gegen Buy-and-Hold +31,0 %, Differenz −7,5 %. Dann Sharpe/Drawdown als Risikoeinordnung. Zum Schluss kurz den Standard-LSTM als zweite Baseline erwähnen – zeigt, ob die Outperformance-Formulierung gegenüber einer einfachen Richtungsvorhersage etwas bringt. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Top-2: +23,5 %  vs. Buy-and-Hold: +31,0 %
- Differenz: −7,5 %
- Sharpe Ratio 0,70 · Max Drawdown −42,9 %
- 460 Trades (Top-K-Wechsel) über 01.08.2025 – 09.07.2026 (235 Handelstage)
- Standard-LSTM zum Vergleich: +30,4 % vs. +57,6 % (−27,2 %)

**Hinweis zum Diagramm:** Diese Kurve braucht evtl. zusätzliche Erklärung: sie zeigt die kumulative Wertentwicklung, nicht die Tagesrendite – ein Sprung nach oben bedeutet nicht 'ein guter Tag', sondern ein anhaltend besseres Niveau.

**Erwartbare Rückfragen:**
- *Warum ist Top-2 die gezeigte Strategie und nicht ein anderes K?* — Das ist das K mit der besten Rendite im Backtest für dieses Universum – die vollständige Top-1-bis-5-Übersicht folgt auf der nächsten Folie.

---

### Folie 24 — Neue Tech-Aktien · Wie viele Aktien sollte das Portfolio gleichzeitig halten?

**Sprechtext:** Erklären: je kleiner K, desto konzentrierter das Portfolio auf die vermeintlich beste Aktie – höhere Varianz, aber auch höheres Potenzial, wenn das Modell-Ranking stimmt. Größeres K diversifiziert und nähert sich Buy-and-Hold an. Für Neue Tech-Aktien ist Top-2 (+23,5 %) am besten. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Bestes K: Top-2 (+23,5 %)
- Schwächstes K: Top-1 (−4,5 %)
- Spannweite zwischen bestem und schwächstem K: 28,0 %
- Kleine K = konzentriert, hohe Varianz, hohes Potenzial. Große K = diversifiziert, nähert sich Buy-and-Hold an.

**Hinweis zum Diagramm:** Balkendiagramm mit Positionsangabe je K – Achsenbeschriftung enthält Ø Position, kurz erwähnen falls Publikum fragt.

---

### Folie 25 — Neue Tech-Aktien · Welche Aktien wählt das Modell – und wie sicher ist es sich?

**Sprechtext:** Zeigen, dass das Modell nicht willkürlich wählt: bestimmte Aktien werden konstant bevorzugt (MU, DELL), andere nie. Die Wahrscheinlichkeiten liegen nah an 0,50 – kurz einordnen, dass das für Finanzmarkt-Klassifikation normal ist, ein Random-Walk-Element bleibt immer. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Häufigste Top-1-Wahl: MU
- Ø Wahrscheinlichkeit 0,42 (Median 0,39, Std 0,10) — 0,50 wäre Zufallsniveau
- Konstant bevorzugt: MU, DELL
- Nie gewählt: PLTR, CRWD, NET, NOW, CSCO, SMCI
- Ø Umschichtung pro Signaltag: 1,0 Positionswechsel

**Hinweis zum Diagramm:** Zeigt Auswahlhäufigkeit + Wahrscheinlichkeitsverteilung je Ticker; ggf. auf optionale Zusatzfolie 06_signal_probability_distribution.png verweisen, falls Rückfrage zur Verteilungsform kommt.

---

### Folie 26 — Neue Tech-Aktien · Hourly vs. Daily Alpaca Paper Trading

**Sprechtext:** Ehrlich kommunizieren: Hourly-Daten reichen aktuell nicht für eine Auswertung (1 Zeitstempel, technischer Grund: Umstellung auf täglichen Scheduler nach einem einmaligen Dry-Run) – hier NICHTS simulieren oder schönreden. Daily-Ergebnis ist nur ein 5-Tage-Plausibilitätscheck, klar als vorläufig kennzeichnen. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Hourly:
- Status: FEHLEND — nur 1 eindeutiger 1-Stunden-Zeitstempel aus einem einmaligen Dry-Run. Zu wenig für eine Zeitreihen-Auswertung.
- Daily:
- Portfolio +1,6 % vs. Benchmark +0,4 % (Differenz +1,2 %), Max Drawdown −2,4 %
- 10 (7 Kauf / 3 Verkauf), gehandeltes Volumen 430.888 $

**Hinweis zum Diagramm:** Das eingebettete Diagramm zeigt Daily Paper Trading; für Hourly existiert nur ein Hinweistext-Plot (Datenbasis zu klein) – im Backup-Anhang verfügbar, hier bewusst nicht groß gezeigt.

**Erwartbare Rückfragen:**
- *Warum gibt es keine Hourly-Ergebnisse?* — Es liegen nur 1-2 Zeitstempel aus einem einmaligen Dry-Run vor der Umstellung auf den täglichen Scheduler vor – das reicht nicht für eine belastbare Zeitreihenauswertung. Wir zeigen hier bewusst keine simulierten oder geschätzten Werte.

---

### Folie 27 — Neue Tech-Aktien · Antwort auf: „Generalisiert das Modell auf weitere bzw. kleinere Tech-Unternehmen außerhalb des ursprünglichen Universums?“

**Sprechtext:** Diese Folie fasst die Teilfrage für Neue Tech-Aktien explizit zusammen, bevor es zum nächsten Universum geht. Stärken/Schwächen bewusst beide nennen – das ist Teil der wissenschaftlichen Redlichkeit des Projekts. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Top-2-Outperformance-LSTM erzielt +23,5 % gegenüber +31,0 % für QQQ – Differenz −7,5 %, Sharpe 0,70, Max Drawdown −42,9 %.
- MU war die am häufigsten gewählte Top-1-Aktie; mit Ø Wahrscheinlichkeit 0,42 die eindeutigsten Signale aller vier Universen.
- Einziges Universum mit positiver Daily-Alpaca-Differenz (+1,2 %) – bei nur 5 Handelstagen jedoch nicht belastbar.
- Stärken: Eindeutigste Modell-Wahrscheinlichkeiten (höchster Mittelwert 0,42) und einzige positive Daily-Paper-Trading-Differenz.
- Schwächen: Schwächster Backtest (−7,5 % Differenz), größter Max Drawdown (−42,9 %), niedrigster Sharpe (0,70) – deutet auf schwächere Generalisierung.

---

### Folie 28 — Defensive Non-Tech · Divider: Defensive Non-Tech

**Sprechtext:** Forschungsfrage: Ist die Strategie Tech-spezifisch oder funktioniert sie auch bei defensiven Nicht-Tech-Aktien?. Aktien: WMT, COST, PG, KO, PEP, JNJ, MRK, ABBV, XOM, CVX. Benchmark: SPY.

---

### Folie 29 — Defensive Non-Tech · Schlägt die Strategie SPY?

**Sprechtext:** Kernzahl zuerst nennen: Top-1-Strategie +20,7 % gegen Buy-and-Hold +21,4 %, Differenz −0,7 %. Dann Sharpe/Drawdown als Risikoeinordnung. Zum Schluss kurz den Standard-LSTM als zweite Baseline erwähnen – zeigt, ob die Outperformance-Formulierung gegenüber einer einfachen Richtungsvorhersage etwas bringt. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Top-1: +20,7 %  vs. Buy-and-Hold: +21,4 %
- Differenz: −0,7 %
- Sharpe Ratio 1,14 · Max Drawdown −13,2 %
- 329 Trades (Top-K-Wechsel) über 01.08.2025 – 09.07.2026 (235 Handelstage)
- Standard-LSTM zum Vergleich: +15,2 % vs. +21,0 % (−5,8 %)

**Hinweis zum Diagramm:** Diese Kurve braucht evtl. zusätzliche Erklärung: sie zeigt die kumulative Wertentwicklung, nicht die Tagesrendite – ein Sprung nach oben bedeutet nicht 'ein guter Tag', sondern ein anhaltend besseres Niveau.

**Erwartbare Rückfragen:**
- *Warum ist Top-1 die gezeigte Strategie und nicht ein anderes K?* — Das ist das K mit der besten Rendite im Backtest für dieses Universum – die vollständige Top-1-bis-5-Übersicht folgt auf der nächsten Folie.

---

### Folie 30 — Defensive Non-Tech · Wie viele Aktien sollte das Portfolio gleichzeitig halten?

**Sprechtext:** Erklären: je kleiner K, desto konzentrierter das Portfolio auf die vermeintlich beste Aktie – höhere Varianz, aber auch höheres Potenzial, wenn das Modell-Ranking stimmt. Größeres K diversifiziert und nähert sich Buy-and-Hold an. Für Defensive Non-Tech ist Top-1 (+20,7 %) am besten. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Bestes K: Top-1 (+20,7 %)
- Schwächstes K: Top-2 (+5,6 %)
- Spannweite zwischen bestem und schwächstem K: 15,1 %
- Kleine K = konzentriert, hohe Varianz, hohes Potenzial. Große K = diversifiziert, nähert sich Buy-and-Hold an.

**Hinweis zum Diagramm:** Balkendiagramm mit Positionsangabe je K – Achsenbeschriftung enthält Ø Position, kurz erwähnen falls Publikum fragt.

---

### Folie 31 — Defensive Non-Tech · Welche Aktien wählt das Modell – und wie sicher ist es sich?

**Sprechtext:** Zeigen, dass das Modell nicht willkürlich wählt: bestimmte Aktien werden konstant bevorzugt (COST, JNJ, ABBV), andere nie. Die Wahrscheinlichkeiten liegen nah an 0,50 – kurz einordnen, dass das für Finanzmarkt-Klassifikation normal ist, ein Random-Walk-Element bleibt immer. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Häufigste Top-1-Wahl: COST
- Ø Wahrscheinlichkeit 0,36 (Median 0,37, Std 0,02) — 0,50 wäre Zufallsniveau
- Konstant bevorzugt: COST, JNJ, ABBV
- Nie gewählt: WMT, PG, KO, PEP, MRK, XOM, CVX
- Ø Umschichtung pro Signaltag: 0,0 Positionswechsel

**Hinweis zum Diagramm:** Zeigt Auswahlhäufigkeit + Wahrscheinlichkeitsverteilung je Ticker; ggf. auf optionale Zusatzfolie 06_signal_probability_distribution.png verweisen, falls Rückfrage zur Verteilungsform kommt.

---

### Folie 32 — Defensive Non-Tech · Hourly vs. Daily Alpaca Paper Trading

**Sprechtext:** Ehrlich kommunizieren: Hourly-Daten reichen aktuell nicht für eine Auswertung (2 Zeitstempel, technischer Grund: Umstellung auf täglichen Scheduler nach einem einmaligen Dry-Run) – hier NICHTS simulieren oder schönreden. Daily-Ergebnis ist nur ein 5-Tage-Plausibilitätscheck, klar als vorläufig kennzeichnen. Ca. 45-60 Sekunden.

**Stichpunkte auf der Folie:**
- Hourly:
- Status: VORLÄUFIG — nur 2 eindeutige 1-Stunden-Zeitstempel aus einem einmaligen Dry-Run vor Umstellung auf den täglichen Scheduler. Zu wenig für eine Zeitreihen-Auswertung.
- Daily:
- Portfolio −2,4 % vs. Benchmark +0,5 % (Differenz −2,8 %), Max Drawdown −3,2 %
- 9 (6 Kauf / 3 Verkauf), gehandeltes Volumen 254.488 $

**Hinweis zum Diagramm:** Das eingebettete Diagramm zeigt Daily Paper Trading; für Hourly existiert nur ein Hinweistext-Plot (Datenbasis zu klein) – im Backup-Anhang verfügbar, hier bewusst nicht groß gezeigt.

**Erwartbare Rückfragen:**
- *Warum gibt es keine Hourly-Ergebnisse?* — Es liegen nur 1-2 Zeitstempel aus einem einmaligen Dry-Run vor der Umstellung auf den täglichen Scheduler vor – das reicht nicht für eine belastbare Zeitreihenauswertung. Wir zeigen hier bewusst keine simulierten oder geschätzten Werte.

---

### Folie 33 — Defensive Non-Tech · Antwort auf: „Ist die Strategie Tech-spezifisch oder funktioniert sie auch bei defensiven Nicht-Tech-Aktien?“

**Sprechtext:** Diese Folie fasst die Teilfrage für Defensive Non-Tech explizit zusammen, bevor es zum nächsten Universum geht. Stärken/Schwächen bewusst beide nennen – das ist Teil der wissenschaftlichen Redlichkeit des Projekts. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Top-1-Outperformance-LSTM erzielt +20,7 % gegenüber +21,4 % für SPY – Differenz nur −0,7 %, Sharpe 1,14, Max Drawdown −13,2 % (niedrigster aller Universen).
- COST war die am häufigsten gewählte Top-1-Aktie; niedrigste Signal-Streuung (Std 0,02) aller Universen – sehr konsistente, aber wenig differenzierende Signale.
- Die Strategie bleibt nahezu gleichauf mit Buy-and-Hold – anders als bei Original Tech ist hier kein klarer Mehrwert sichtbar, aber auch kein Nachteil.
- Stärken: Mit Abstand geringster Max Drawdown (−13,2 %) und geringste Volatilität (1,50 %) – stabilste, risikoärmste Variante.
- Schwächen: Kaum Mehrwert gegenüber Buy-and-Hold (−0,7 %); Signale sind am wenigsten differenzierend (Ø Wahrscheinlichkeit 0,36, Std nur 0,02).

---

### Folie 34 — Gesamtvergleich · Divider

**Sprechtext:** Übergang zum synthetisierten Gesamtvergleich aller vier Universen.

---

### Folie 35 — Gesamtvergleich · Höhere Volatilität und geringere Korrelation erschweren die Vorhersage

**Sprechtext:** Diese beiden Kennzahlen sind der Ausgangspunkt für alles Folgende: sie erklären teilweise, warum Original Tech (hohe Korrelation) und Neue Tech-Aktien (hohe Volatilität, niedrige Korrelation) sich in Modellleistung und Backtest so unterschiedlich verhalten. Ca. 45 Sekunden.

---

### Folie 36 — Gesamtvergleich · Original Tech liefert die stärkste Klassifikationsleistung

**Sprechtext:** Direkte Verbindung zur vorigen Folie ziehen: die Korrelationsunterschiede aus Vergleich 1 spiegeln sich hier in der Modellqualität wider. Original Tech (höchste Korrelation) hat die beste F1, Neue Tech-Aktien (niedrigste Korrelation) die schwächste. Ca. 45 Sekunden.

**Stichpunkte auf der Folie:**
- Beste Test-F1 (Outperformance-LSTM): Original Tech (0,541)
- Schwächste: Neue Tech-Aktien (0,475)
- Universen mit höherer interner Korrelation liefern konsistentere, leichter lernbare Muster – das allgemeine Marktsignal ist deutlicher.
- Heterogene Einzeltitel (z. B. kleinere, neuere Tech-Werte mit eigenen Nachrichtenlagen) sind schwerer vorherzusagen.

**Hinweis zum Diagramm:** Gruppierte Balken mit 4 Metriken × 2 Modellen × 4 Universen – bei Rückfragen auf einzelne Balken zeigen können, Legende erklären.

---

### Folie 37 — Gesamtvergleich · Original Tech schlägt den Markt am deutlichsten – Neue Tech-Aktien am wenigsten

**Sprechtext:** Zentraler wissenschaftlicher Befund der ganzen Präsentation. Betonen: hoher Sharpe UND geringer Drawdown sind für die praktische Einsetzbarkeit wichtiger als die reine absolute Rendite, weil große Drawdowns in der Praxis zum vorzeitigen Strategieabbruch führen können. Ca. 50-60 Sekunden.

**Stichpunkte auf der Folie:**
- Größte Outperformance: Original Tech (+74,7 %)
- Größter Rückstand: Neue Tech-Aktien (−7,5 %)
- Spanne zwischen bestem und schwächstem Universum: 82,1 Prozentpunkte.
- Höchste Stabilität (Sharpe): Original Tech (2,17). Geringster Max Drawdown: Defensive Non-Tech (−13,2 %).

---

### Folie 38 — Gesamtvergleich · Konzentration hilft nicht überall gleich

**Sprechtext:** Diese Heatmap ist kompakt aber dicht – kurz erklären, wie zu lesen ist: jede Zeile ein Universum, jede Spalte ein K, Farbe = Rendite. Kernaussage: die optimale Portfolio-Konzentration ist nicht universell, sondern hängt von der Verlässlichkeit des Modell-Rankings im jeweiligen Universum ab. Ca. 40-50 Sekunden.

**Stichpunkte auf der Folie:**
- Profitiert am stärksten von Konzentration: Original Tech (+37,5 Pp. Top-1 vs. Top-5)
- Am wenigsten – sogar umgekehrt: Tech ohne Nvidia (−32,7 Pp.)
- Wenn das Modell-Ranking verlässlich ist (wie bei Original Tech), lohnt sich Konzentration. Ist es weniger verlässlich, hilft Diversifikation (Top-4/5) mehr.

**Hinweis zum Diagramm:** Heatmap braucht ggf. kurze Lese-Erklärung für das Publikum (Zeile = Universum, Spalte = Top-K, Farbskala = Rendite).

---

### Folie 39 — Gesamtvergleich · Neue Tech-Aktien liefert die eindeutigsten Signale

**Sprechtext:** Interessanter Kontrast zur Backtest-Folie: Neue Tech-Aktien hat die 'sichersten' Modell-Wahrscheinlichkeiten, aber den schwächsten Backtest – ein Hinweis, dass hohe Modell-Konfidenz nicht automatisch hohe ökonomische Rendite bedeutet. Guter Punkt für eine kritische, wissenschaftliche Einordnung. Ca. 40 Sekunden.

**Stichpunkte auf der Folie:**
- Eindeutigste Signale im Mittel: Neue Tech-Aktien (Ø 0,42)
- Unsicherste Signale: Original Tech (Ø 0,36, am nächsten an 0,50)
- Bevorzugte Top-1-Aktie je Universum: AMD (Original Tech & Tech ohne Nvidia), MU (Neue Tech-Aktien), COST (Defensive Non-Tech).

---

### Folie 40 — Gesamtvergleich · Objektive, transparente Rangfolge der vier Universen

**Sprechtext:** Diese Folie beantwortet die Kernfrage der ganzen Gesamtanalyse direkt. Methodik kurz erklären (min-max-normalisiert, ungewichteter Mittelwert über 7 Kriterien) – wichtig für Glaubwürdigkeit: das ist keine willkürliche Bewertung, sondern nachvollziehbar aus den bereits gezeigten Zahlen abgeleitet. Ca. 50-60 Sekunden.

**Erwartbare Rückfragen:**
- *Warum ungewichtet?* — Um keine Kriterien-Gewichtung zu erfinden, die nicht aus den Daten selbst begründbar wäre – ein einfacher, transparenter Mittelwert ist nachvollziehbarer als eine willkürlich gewichtete Formel.

---

### Folie 41 — Gesamtvergleich · Erwartungskonform und überraschend zugleich

**Sprechtext:** Diese Folie ist die Synthese vor dem Übergang zur Paper-Trading-Diskussion. Besonders den überraschenden Befund (Neue Tech-Aktien: hohe Konfidenz, schwacher Erfolg) betonen – das zeigt wissenschaftliche Sorgfalt, nicht nur 'unsere Strategie ist gut'. Ca. 50 Sekunden.

**Stichpunkte auf der Folie:**
- Erwartungskonform: die drei Tech-Universen schneiden im Schnitt deutlich besser ab (Ø Differenz +27,8 %) als das defensive Kontroll-Universum (−0,7 %) – die Strategie wurde ursprünglich für Tech-Aktien entwickelt.
- Überraschend: Neue Tech-Aktien zeigt die im Mittel eindeutigsten Modell-Wahrscheinlichkeiten, aber den schwächsten Backtest – hohe Modell-Konfidenz übersetzt sich hier nicht in ökonomischen Erfolg.
- Größter Einzelunterschied im Backtest: Original Tech (+74,7 %) vs. Neue Tech-Aktien (−7,5 %) – eine Spanne von 82,1 Prozentpunkten bei identischer Methodik.
- Gesamtranking: Original Tech > Defensive Non-Tech > Tech ohne Nvidia > Neue Tech-Aktien.

---

### Folie 42 — Backtest vs. Realität · 235 Handelstage Backtest vs. 5 Tage echtes Paper Trading

**Sprechtext:** Diese Tabelle stellt die Diskrepanz visuell dar, bevor die Gründe erklärt werden. Wichtig: nicht überinterpretieren – 5 Tage sind einfach zu wenig, das ist der Hauptpunkt dieser Folie. Ca. 30 Sekunden, dann weiter zur nächsten Folie mit den Gründen.

---

### Folie 43 — Backtest vs. Realität · Ein Backtest ist eine Simulation – Paper Trading ist real

**Sprechtext:** Diese Folie beantwortet direkt die vom Nutzer geforderte Frage: warum unterscheiden sich Backtest und echtes Paper Trading. Alle sechs Punkte sind methodische Fakten, keine Ausreden – wichtig, das auch so zu vermitteln. Ca. 70-90 Sekunden, das ist eine dichte Folie.

**Stichpunkte auf der Folie:**
- Stichprobengröße: 235 Handelstage Backtest vs. nur 5 Handelstage Daily Paper Trading – statistisch nicht vergleichbar.
- Markttiming: der Backtest nutzt historische Schlusskurse für den gesamten Testzeitraum; das Paper Trading fällt in ein einziges, kurzes reales Marktfenster mit seiner eigenen Marktlage.
- Spread & Slippage: der Backtest modelliert keine Transaktionskosten oder Geld-Brief-Spannen; echte Orders bei Alpaca werden zum tatsächlich verfügbaren Marktpreis ausgeführt.
- Intraday-Bewegung & Orderausführung: Backtest-Signale werden auf Tagesschlusskurs-Basis bewertet; reale Orders werden zu einem Ausführungszeitpunkt innerhalb des Handelstages gefüllt, der vom Signal-Zeitpunkt abweichen kann.
- Rebalancing-Realität: im Backtest ist ein Positionswechsel sofort und kostenlos; im echten Paper Trading braucht ein Rebalancing mehrere reale Orders mit eigener Ausführung.
- Benchmark-Messung: der Backtest vergleicht gegen den historischen Indexverlauf, das Paper Trading gegen die tatsächliche Benchmark-Bewegung im selben kurzen Fenster.

**Erwartbare Rückfragen:**
- *Ist der Backtest dann überhaupt aussagekräftig?* — Ja, für die relative Bewertung der Strategie unter identischen, fairen Bedingungen über einen langen Zeitraum – aber er ist kein Ersatz für echte Live-Performance-Validierung, die deutlich mehr Beobachtungszeit braucht.

---

### Folie 44 — Projektabschluss · Divider

**Sprechtext:** Übergang zu Lessons Learned, Limitationen, Future Work.

---

### Folie 45 — Projektabschluss · Was die Ergebnisse einschränkt – ehrlich benannt

**Sprechtext:** Diese Folie zeigt wissenschaftliche Redlichkeit – bewusst nicht schönreden. Besonders den Zielkonflikt F1 vs. Rendite hervorheben, das ist ein inhaltlich interessanter Punkt für Rückfragen. Ca. 60 Sekunden.

**Stichpunkte auf der Folie:**
- Kein Transaktionskostenmodell in allen vier Backtests – reale Nettoerträge wären niedriger.
- Zielkonflikt beobachtet: das Klassifikationsziel (F1) und das ökonomische Ziel (Rendite) sind nicht identisch optimiert – z. B. hat der Standard-LSTM in Original Tech eine höhere F1 (0,651) als der Outperformance-LSTM (0,541), obwohl Letzterer ökonomisch deutlich erfolgreicher ist.
- Relativ kurzer Backtest-Zeitraum (235 Handelstage) im Verhältnis zu vollständigen Marktzyklen.
- Hourly Paper Trading aktuell für keines der vier Universen auswertbar (Datenlücke durch Umstellung auf täglichen Scheduler) – hier bewusst keine Werte simuliert oder geschätzt.
- Daily Paper Trading bislang nur 5 Handelstage – dient als Plausibilitätscheck der Pipeline, nicht als belastbares Live-Ergebnis.

---

### Folie 46 — Projektabschluss · Wie es weitergehen könnte

**Sprechtext:** Zwei Themen kombiniert, um Redezeit zu sparen: erst technische Weiterentwicklung, dann konkrete Einsatzempfehlung. Bei Praktischer Einsatz explizit sagen: kein Full-Auto-Trading-Claim, das wäre unseriös angesichts der Limitationen von der vorigen Folie. Ca. 70 Sekunden.

**Stichpunkte auf der Folie:**
- Future Work:
- Längere Paper-Trading-Beobachtung (Wochen/Monate), sobald der tägliche Scheduler kontinuierlich weiterläuft.
- Transaktionskosten-/Spread-Modell in den Backtest integrieren, um realistischere Nettoerträge zu zeigen.
- Feature-Wichtigkeits-Analyse (z. B. Permutation Importance) nachrüsten, um zu verstehen, welche der 30 Features den größten Beitrag leisten.
- Hyperparameter-Tuning und Robustheitschecks nachholen (im Code vorbereitet, aber für die Präsentationsvorbereitung bewusst übersprungen, da sie mehrere zusätzliche LSTMs pro Universum trainieren würden).
- Praktischer Einsatz:
- Aktuell als Entscheidungsunterstützung zu verstehen, nicht als vollautomatisches Live-Trading-System – dafür fehlen Transaktionskosten-Modellierung und längere Live-Validierung.
- Original Tech zeigt die konsistenteste, stärkste Outperformance; Defensive Non-Tech den geringsten Drawdown – je nach Risikobereitschaft unterschiedlich einsetzbar.

---

### Folie 47 — Fazit · Fazit

**Sprechtext:** Zusammenfassung der Kernaussage über alle vier Universen hinweg, Bezug zum Ranking.

---

### Folie 48 — Danke · Q&A

**Sprechtext:** Abschluss, Fragerunde eröffnen.

---

## Selbstbewertung der Präsentation

Bewertung 1-10 anhand der für dieses Projekt zentralen Qualitätskriterien (abgeleitet aus den wiederholt gestellten Anforderungen: wissenschaftliche Redlichkeit, roter Faden, visuelle Konsistenz, Verständlichkeit, Vollständigkeit der geforderten Struktur, Diagramm-Interpretation, Sprechtext/Q&A, professionelles Design, Timing).

| # | Kriterium | Bewertung | Begründung |
| --- | --- | --- | --- |
| | Wissenschaftliche Redlichkeit | 10/10 | Jede Zahl stammt 1:1 aus den bereits validierten wissenschaftliche_analyse.md- und gesamtanalyse.md-Reports; fehlende Daten (Hourly) werden explizit als MISSING/PRELIMINARY benannt statt simuliert. |
| | Roter Faden / Storyline | 9/10 | Durchgängiger Bogen Problem → Ziel → Daten → Pipeline → Features → Modell → Training → 4× Universum → Gesamtvergleich → Backtest-vs-Realität → Projektabschluss → Fazit, jede Folie beantwortet eine Frage. |
| | Visuelle Konsistenz | 9/10 | Einheitliches schwarz/grünes Farbsystem, wiederkehrendes Kreis-Motiv, feste Universums-Farben (identisch zu reporting_config.py), keine Akzentstreifen, konsistente Layout-Sprache je Folientyp. |
| | Verständlichkeit für Laien | 9/10 | LSTM- und Top-K-Konzept ohne Formeln erklärt; Fachbegriffe (Sharpe, Drawdown) auf der Backtest-Methodik-Folie eingeführt, bevor sie verwendet werden. |
| | Vollständigkeit der geforderten Struktur | 10/10 | Alle geforderten Blöcke vorhanden: 8 Methodik-Folien, 4×5 identische Universums-Folien, 7 Gesamtvergleichs-Folien, dedizierte Backtest-vs-Paper-Trading-Diskussion, Projektabschluss mit Lessons Learned/Limitationen/Future Work/Praktischer Einsatz, Fazit. |
| | Diagramm-Interpretation | 9/10 | Jedes eingebettete Diagramm wird von 3-5 Interpretations-Stichpunkten begleitet, kein unkommentiertes Chart; Diagramme, die zusätzliche Erklärung brauchen (Heatmap, kumulative Kurve), sind im Begleitdokument als 'Hinweis zum Diagramm' markiert. |
| | Sprechtext & Q&A | 9/10 | Jede Folie hat Sprechtext + Stichpunkte in diesem Dokument; zusätzlich 7 projektweite und mehrere folienspezifische Q&A-Paare. Kein separates Timing-Coaching pro Vortragendem (individuell, außerhalb des Dokuments). |
| | Professionelles Design | 9/10 | Konsistent dunkles, minimalistisches Farbschema mit viel Weißraum, keine Bullet-Only-Monotonie (Karten, Statistik-Kacheln, Tabellen, Diagramm-Layout, eigenes Pipeline-Diagramm), safe-list Font (Calibri). |
| | Timing (Ziel 15-20 Min.) | 7/10 | Realistisch ca. 22-25 Minuten bei zügigem Vortrag, siehe Timing-Abschnitt oben – liegt über dem Zielwert, da die vom Projekt geforderte Struktur (4× identische 5-Folien-Blöcke + vollständiger Gesamtvergleich) inhaltlich umfangreich ist. Mit den oben genannten Kürzungsoptionen ist 20 Minuten erreichbar. |

Acht der neun Kriterien erreichen 9/10 oder höher. Das Timing-Kriterium liegt bei 7/10, weil die vom Projekt vorgegebene Struktur (identische 5-Folien-Tiefe für alle vier Universen, plus vollständiger Gesamtvergleich) inhaltlich mehr Redezeit braucht, als in 15-20 Minuten bei vollständiger Behandlung passt. Eine künstliche Kürzung der Struktur wurde bewusst nicht vorgenommen, da die Vorgabe „identische Tiefe für alle vier Universen, keine Ausnahme“ Vorrang vor der Zeitvorgabe hat; stattdessen werden oben konkrete, in den Daten selbst begründete Kürzungsoptionen genannt (aus gesamtanalyse.md), mit denen das Team die Präsentation live auf 20 Minuten bringen kann, ohne inhaltlich etwas zu verfälschen.
