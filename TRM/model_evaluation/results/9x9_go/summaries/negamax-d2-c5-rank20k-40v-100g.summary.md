# Riepilogo test TRM vs KataGo — 9x9

- **Stato:** COMPLETATO
- **Inizio (UTC):** 2026-11-15 13:32:01
- **Fine (UTC):** 2026-11-15 13:36:18
- **Durata:** 4.3 minuti (257.4 secondi)
- **Avversario:** KataGo 1.18.2+kata9x9-b18c384nbt-s6603M+b18c384nbt-humanv0
- **Modalità TRM:** negamax

## Parametri del test

Comando eseguito:
```bash
python3 model_evaluation/go_match.py --board-size 9 --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago --katago-config model_evaluation/katago-local/engine/default_gtp.cfg --katago-model model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz --human-profile rank_20k --games 100 --max-visits 40 --seed 42 --pass-logit-penalty 3.5 --pass-penalty-before-stones 16 --search negamax --search-depth 2 --search-candidates 5 --output model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.csv
```

La tabella riporta gli argomenti CLI (compresi i valori di default) e i parametri effettivi risolti dallo script.

| Argomento CLI | Valore |
| --- | --- |
| --board-size | 9 |
| --checkpoint | TRM/checkpoints/best_model.pt |
| --rules | TRM/model_evaluation/config/rules.json |
| --katago | TRM/model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago |
| --katago-config | TRM/model_evaluation/katago-local/engine/default_gtp.cfg |
| --katago-model | TRM/model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz |
| --katago-human-model | TRM/model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz |
| --human-profile | rank_20k |
| --human-temperature-mode | — |
| --games | 100 |
| --max-moves | — |
| --search | negamax |
| --search-depth | 2 |
| --search-candidates | 5 |
| --max-visits | 40 |
| --gtp-timeout | 600.0 |
| --device | auto |
| --pass-logit-penalty | 3.5 |
| --pass-penalty-before-stones | 16 |
| --seed | 42 |
| --output | TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.csv |
| --report-output | TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.report.json |
| --summary-output | TRM/model_evaluation/results/9x9_go/summaries/negamax-d2-c5-rank20k-40v-100g.summary.md |

### Parametri effettivi e file generati

| Parametro | Valore |
| --- | --- |
| Checkpoint TRM | TRM/checkpoints/best_model.pt |
| Device effettivo | cuda |
| Komi | 7.5 |
| Limite mosse effettivo | 324 |
| Regole GTP effettive | {"friendlyPassOk": true, "hasButton": false, "ko": "POSITIONAL", "scoring": "AREA", "suicide": false, "tax": "NONE", "whiteHandicapBonus": "0"} |
| Profilo HumanSL effettivo | rank_20k |
| Candidati Negamax effettivi | 5 |
| Profondità Negamax effettiva | 2 |
| CSV partite | TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.csv |
| Full report JSON | TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.report.json |
| Riepilogo Markdown | TRM/model_evaluation/results/9x9_go/summaries/negamax-d2-c5-rank20k-40v-100g.summary.md |

## Risultati complessivi

| Indicatore | Risultato |
| --- | --- |
| Partite richieste | 100 |
| Partite registrate | 100 |
| Partite completate | 100 |
| Partite incomplete | 0 |
| Partite non iniziate | 0 |
| Vittorie TRM | 58 |
| Pareggi | 0 |
| Sconfitte TRM (vittorie KataGo) | 42 |
| Win rate TRM | 58.0% |
| Score rate TRM (vittoria=1, pareggio=0.5) | 58.0% |
| Intervallo 95% win rate (Wilson) | 48.2%–67.2% |
| Gap medio TRM | -24.8 pt |
| Gap mediano TRM | +1.5 pt |
| Gap medio assoluto | 33.3 pt |
| Partite con punteggio numerico | 100 |
| Gap a favore TRM / a favore KataGo / pari | 58 / 42 / 0 |
| Mosse medie per partita registrata | 98.7 |
| PASS giocati dal TRM | 122 |
| PASS prima di 16 pietre | 0 |
| Decisioni con penalità PASS attiva | 800 |
| Terminazioni | two_passes: 100 |

*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*

## Risultati per colore del TRM

| Colore TRM | Giocate | Completate | V-P-S | Win rate | Gap medio |
| --- | --- | --- | --- | --- | --- |
| Nero | 50 | 50 | 18-0-32 | 36.0% | -46.5 pt |
| Bianco | 50 | 50 | 40-0-10 | 80.0% | -3.1 pt |

## Statistiche di ricerca

| Indicatore | Totale |
| --- | --- |
| Nodi esplorati | 123573 |
| Foglie valutate | 92770 |
| Cutoff | 15682 |
| Chiamate alla policy | 30803 |
| Tempo totale ricerca | 79.89 s |

*Le statistiche di ricerca sono specifiche del Negamax; con `--search policy` restano a zero.*

## Risultato di ogni partita

| # | Colore TRM | Esito TRM | Vincitore | Terminazione | Mosse | Score B–W | Gap TRM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Nero | Vittoria | Nero | Due pass consecutivi | 72 | 43.0–42.5 | +0.5 pt |
| 2 | Bianco | Vittoria | Bianco | Due pass consecutivi | 78 | 39.0–44.5 | +5.5 pt |
| 3 | Nero | Sconfitta | Bianco | Due pass consecutivi | 144 | 7.0–76.5 | -69.5 pt |
| 4 | Bianco | Vittoria | Bianco | Due pass consecutivi | 51 | 30.0–55.5 | +25.5 pt |
| 5 | Nero | Vittoria | Nero | Due pass consecutivi | 63 | 46.0–40.5 | +5.5 pt |
| 6 | Bianco | Vittoria | Bianco | Due pass consecutivi | 74 | 43.0–44.5 | +1.5 pt |
| 7 | Nero | Vittoria | Nero | Due pass consecutivi | 56 | 45.0–37.5 | +7.5 pt |
| 8 | Bianco | Vittoria | Bianco | Due pass consecutivi | 68 | 40.0–42.5 | +2.5 pt |
| 9 | Nero | Sconfitta | Bianco | Due pass consecutivi | 146 | 8.0–74.5 | -66.5 pt |
| 10 | Bianco | Vittoria | Bianco | Due pass consecutivi | 66 | 36.0–46.5 | +10.5 pt |
| 11 | Nero | Sconfitta | Bianco | Due pass consecutivi | 170 | 4.0–80.5 | -76.5 pt |
| 12 | Bianco | Vittoria | Bianco | Due pass consecutivi | 63 | 41.0–42.5 | +1.5 pt |
| 13 | Nero | Sconfitta | Bianco | Due pass consecutivi | 142 | 5.0–78.5 | -73.5 pt |
| 14 | Bianco | Vittoria | Bianco | Due pass consecutivi | 87 | 41.0–45.5 | +4.5 pt |
| 15 | Nero | Sconfitta | Bianco | Due pass consecutivi | 146 | 0.0–88.5 | -88.5 pt |
| 16 | Bianco | Vittoria | Bianco | Due pass consecutivi | 73 | 40.0–44.5 | +4.5 pt |
| 17 | Nero | Sconfitta | Bianco | Due pass consecutivi | 150 | 7.0–78.5 | -71.5 pt |
| 18 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 37.0–45.5 | +8.5 pt |
| 19 | Nero | Sconfitta | Bianco | Due pass consecutivi | 152 | 0.0–88.5 | -88.5 pt |
| 20 | Bianco | Vittoria | Bianco | Due pass consecutivi | 68 | 40.0–42.5 | +2.5 pt |
| 21 | Nero | Vittoria | Nero | Due pass consecutivi | 95 | 44.0–41.5 | +2.5 pt |
| 22 | Bianco | Vittoria | Bianco | Due pass consecutivi | 75 | 41.0–46.5 | +5.5 pt |
| 23 | Nero | Sconfitta | Bianco | Due pass consecutivi | 124 | 6.0–76.5 | -70.5 pt |
| 24 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 39.0–45.5 | +6.5 pt |
| 25 | Nero | Sconfitta | Bianco | Due pass consecutivi | 132 | 3.0–82.5 | -79.5 pt |
| 26 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 35.0–50.5 | +15.5 pt |
| 27 | Nero | Vittoria | Nero | Due pass consecutivi | 70 | 44.0–39.5 | +4.5 pt |
| 28 | Bianco | Sconfitta | Nero | Due pass consecutivi | 120 | 67.0–17.5 | -49.5 pt |
| 29 | Nero | Sconfitta | Bianco | Due pass consecutivi | 108 | 10.0–72.5 | -62.5 pt |
| 30 | Bianco | Vittoria | Bianco | Due pass consecutivi | 74 | 43.0–44.5 | +1.5 pt |
| 31 | Nero | Vittoria | Nero | Due pass consecutivi | 113 | 45.0–42.5 | +2.5 pt |
| 32 | Bianco | Vittoria | Bianco | Due pass consecutivi | 77 | 43.0–43.5 | +0.5 pt |
| 33 | Nero | Sconfitta | Bianco | Due pass consecutivi | 144 | 6.0–79.5 | -73.5 pt |
| 34 | Bianco | Vittoria | Bianco | Due pass consecutivi | 58 | 40.0–48.5 | +8.5 pt |
| 35 | Nero | Sconfitta | Bianco | Due pass consecutivi | 141 | 3.0–82.5 | -79.5 pt |
| 36 | Bianco | Vittoria | Bianco | Due pass consecutivi | 61 | 35.0–50.5 | +15.5 pt |
| 37 | Nero | Sconfitta | Bianco | Due pass consecutivi | 150 | 5.0–80.5 | -75.5 pt |
| 38 | Bianco | Vittoria | Bianco | Due pass consecutivi | 62 | 36.0–50.5 | +14.5 pt |
| 39 | Nero | Sconfitta | Bianco | Due pass consecutivi | 134 | 9.0–75.5 | -66.5 pt |
| 40 | Bianco | Vittoria | Bianco | Due pass consecutivi | 55 | 42.0–46.5 | +4.5 pt |
| 41 | Nero | Sconfitta | Bianco | Due pass consecutivi | 148 | 2.0–84.5 | -82.5 pt |
| 42 | Bianco | Sconfitta | Nero | Due pass consecutivi | 135 | 65.0–17.5 | -47.5 pt |
| 43 | Nero | Sconfitta | Bianco | Due pass consecutivi | 132 | 0.0–88.5 | -88.5 pt |
| 44 | Bianco | Sconfitta | Nero | Due pass consecutivi | 126 | 64.0–20.5 | -43.5 pt |
| 45 | Nero | Sconfitta | Bianco | Due pass consecutivi | 133 | 9.0–75.5 | -66.5 pt |
| 46 | Bianco | Sconfitta | Nero | Due pass consecutivi | 126 | 62.0–19.5 | -42.5 pt |
| 47 | Nero | Sconfitta | Bianco | Due pass consecutivi | 112 | 10.0–73.5 | -63.5 pt |
| 48 | Bianco | Vittoria | Bianco | Due pass consecutivi | 49 | 26.0–42.5 | +16.5 pt |
| 49 | Nero | Sconfitta | Bianco | Due pass consecutivi | 150 | 0.0–88.5 | -88.5 pt |
| 50 | Bianco | Vittoria | Bianco | Due pass consecutivi | 49 | 28.0–55.5 | +27.5 pt |
| 51 | Nero | Sconfitta | Bianco | Due pass consecutivi | 162 | 3.0–83.5 | -80.5 pt |
| 52 | Bianco | Vittoria | Bianco | Due pass consecutivi | 87 | 41.0–42.5 | +1.5 pt |
| 53 | Nero | Vittoria | Nero | Due pass consecutivi | 81 | 45.0–42.5 | +2.5 pt |
| 54 | Bianco | Vittoria | Bianco | Due pass consecutivi | 69 | 41.0–47.5 | +6.5 pt |
| 55 | Nero | Vittoria | Nero | Due pass consecutivi | 70 | 44.0–40.5 | +3.5 pt |
| 56 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 42.0–44.5 | +2.5 pt |
| 57 | Nero | Vittoria | Nero | Due pass consecutivi | 71 | 44.0–43.5 | +0.5 pt |
| 58 | Bianco | Vittoria | Bianco | Due pass consecutivi | 71 | 38.0–45.5 | +7.5 pt |
| 59 | Nero | Vittoria | Nero | Due pass consecutivi | 47 | 46.0–39.5 | +6.5 pt |
| 60 | Bianco | Vittoria | Bianco | Due pass consecutivi | 52 | 25.0–34.5 | +9.5 pt |
| 61 | Nero | Sconfitta | Bianco | Due pass consecutivi | 142 | 3.0–82.5 | -79.5 pt |
| 62 | Bianco | Sconfitta | Nero | Due pass consecutivi | 141 | 76.0–10.5 | -65.5 pt |
| 63 | Nero | Sconfitta | Bianco | Due pass consecutivi | 154 | 2.0–84.5 | -82.5 pt |
| 64 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 31.0–44.5 | +13.5 pt |
| 65 | Nero | Vittoria | Nero | Due pass consecutivi | 78 | 43.0–41.5 | +1.5 pt |
| 66 | Bianco | Sconfitta | Nero | Due pass consecutivi | 100 | 55.0–29.5 | -25.5 pt |
| 67 | Nero | Sconfitta | Bianco | Due pass consecutivi | 132 | 6.0–79.5 | -73.5 pt |
| 68 | Bianco | Vittoria | Bianco | Due pass consecutivi | 64 | 37.0–45.5 | +8.5 pt |
| 69 | Nero | Vittoria | Nero | Due pass consecutivi | 113 | 45.0–42.5 | +2.5 pt |
| 70 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 36.0–50.5 | +14.5 pt |
| 71 | Nero | Vittoria | Nero | Due pass consecutivi | 97 | 50.0–38.5 | +11.5 pt |
| 72 | Bianco | Vittoria | Bianco | Due pass consecutivi | 56 | 33.0–55.5 | +22.5 pt |
| 73 | Nero | Sconfitta | Bianco | Due pass consecutivi | 116 | 8.0–77.5 | -69.5 pt |
| 74 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 39.0–47.5 | +8.5 pt |
| 75 | Nero | Vittoria | Nero | Due pass consecutivi | 113 | 45.0–42.5 | +2.5 pt |
| 76 | Bianco | Vittoria | Bianco | Due pass consecutivi | 63 | 39.0–43.5 | +4.5 pt |
| 77 | Nero | Sconfitta | Bianco | Due pass consecutivi | 156 | 3.0–84.5 | -81.5 pt |
| 78 | Bianco | Vittoria | Bianco | Due pass consecutivi | 88 | 39.0–40.5 | +1.5 pt |
| 79 | Nero | Sconfitta | Bianco | Due pass consecutivi | 150 | 3.0–82.5 | -79.5 pt |
| 80 | Bianco | Vittoria | Bianco | Due pass consecutivi | 45 | 22.0–48.5 | +26.5 pt |
| 81 | Nero | Sconfitta | Bianco | Due pass consecutivi | 154 | 7.0–76.5 | -69.5 pt |
| 82 | Bianco | Vittoria | Bianco | Due pass consecutivi | 71 | 40.0–46.5 | +6.5 pt |
| 83 | Nero | Sconfitta | Bianco | Due pass consecutivi | 134 | 9.0–74.5 | -65.5 pt |
| 84 | Bianco | Vittoria | Bianco | Due pass consecutivi | 69 | 41.0–47.5 | +6.5 pt |
| 85 | Nero | Vittoria | Nero | Due pass consecutivi | 48 | 48.0–34.5 | +13.5 pt |
| 86 | Bianco | Sconfitta | Nero | Due pass consecutivi | 137 | 73.0–13.5 | -59.5 pt |
| 87 | Nero | Sconfitta | Bianco | Due pass consecutivi | 124 | 6.0–80.5 | -74.5 pt |
| 88 | Bianco | Vittoria | Bianco | Due pass consecutivi | 72 | 41.0–42.5 | +1.5 pt |
| 89 | Nero | Vittoria | Nero | Due pass consecutivi | 63 | 46.0–40.5 | +5.5 pt |
| 90 | Bianco | Vittoria | Bianco | Due pass consecutivi | 66 | 41.0–44.5 | +3.5 pt |
| 91 | Nero | Sconfitta | Bianco | Due pass consecutivi | 146 | 11.0–72.5 | -61.5 pt |
| 92 | Bianco | Sconfitta | Nero | Due pass consecutivi | 133 | 70.0–14.5 | -55.5 pt |
| 93 | Nero | Vittoria | Nero | Due pass consecutivi | 113 | 45.0–42.5 | +2.5 pt |
| 94 | Bianco | Vittoria | Bianco | Due pass consecutivi | 70 | 39.0–43.5 | +4.5 pt |
| 95 | Nero | Vittoria | Nero | Due pass consecutivi | 113 | 45.0–42.5 | +2.5 pt |
| 96 | Bianco | Sconfitta | Nero | Due pass consecutivi | 116 | 69.0–15.5 | -53.5 pt |
| 97 | Nero | Sconfitta | Bianco | Due pass consecutivi | 128 | 7.0–78.5 | -71.5 pt |
| 98 | Bianco | Sconfitta | Nero | Due pass consecutivi | 126 | 74.0–12.5 | -61.5 pt |
| 99 | Nero | Sconfitta | Bianco | Due pass consecutivi | 134 | 2.0–84.5 | -82.5 pt |
| 100 | Bianco | Vittoria | Bianco | Due pass consecutivi | 63 | 34.0–48.5 | +14.5 pt |

## Altri file

- CSV dettagliato: `TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.csv`
- Full report JSON: `TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.report.json`

