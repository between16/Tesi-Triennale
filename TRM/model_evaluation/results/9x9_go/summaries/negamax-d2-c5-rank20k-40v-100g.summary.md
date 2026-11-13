# Riepilogo test TRM vs KataGo — 9x9

- **Stato:** COMPLETATO
- **Inizio (UTC):** 2026-11-13 20:05:32
- **Fine (UTC):** 2026-11-13 20:09:46
- **Durata:** 4.2 minuti (254.6 secondi)
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
| Vittorie TRM | 64 |
| Pareggi | 0 |
| Sconfitte TRM (vittorie KataGo) | 36 |
| Win rate TRM | 64.0% |
| Score rate TRM (vittoria=1, pareggio=0.5) | 64.0% |
| Intervallo 95% win rate (Wilson) | 54.2%–72.7% |
| Gap medio TRM | -17.1 pt |
| Gap mediano TRM | +2.5 pt |
| Gap medio assoluto | 29.0 pt |
| Partite con punteggio numerico | 100 |
| Gap a favore TRM / a favore KataGo / pari | 64 / 36 / 0 |
| Mosse medie per partita registrata | 94.1 |
| PASS giocati dal TRM | 116 |
| PASS prima di 16 pietre | 0 |
| Decisioni con penalità PASS attiva | 803 |
| Terminazioni | two_passes: 100 |

*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*

## Risultati per colore del TRM

| Colore TRM | Giocate | Completate | V-P-S | Win rate | Gap medio |
| --- | --- | --- | --- | --- | --- |
| Nero | 50 | 50 | 25-0-25 | 50.0% | -31.6 pt |
| Bianco | 50 | 50 | 39-0-11 | 78.0% | -2.7 pt |

## Statistiche di ricerca

| Indicatore | Totale |
| --- | --- |
| Nodi esplorati | 120084 |
| Foglie valutate | 90727 |
| Cutoff | 14646 |
| Chiamate alla policy | 29357 |
| Tempo totale ricerca | 74.94 s |

*Le statistiche di ricerca sono specifiche del Negamax; con `--search policy` restano a zero.*

## Risultato di ogni partita

| # | Colore TRM | Esito TRM | Vincitore | Terminazione | Mosse | Score B–W | Gap TRM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Nero | Sconfitta | Bianco | Due pass consecutivi | 164 | 0.0–88.5 | -88.5 pt |
| 2 | Bianco | Sconfitta | Nero | Due pass consecutivi | 145 | 67.0–15.5 | -51.5 pt |
| 3 | Nero | Sconfitta | Bianco | Due pass consecutivi | 112 | 38.0–48.5 | -10.5 pt |
| 4 | Bianco | Vittoria | Bianco | Due pass consecutivi | 70 | 32.0–56.5 | +24.5 pt |
| 5 | Nero | Vittoria | Nero | Due pass consecutivi | 30 | 47.0–35.5 | +11.5 pt |
| 6 | Bianco | Vittoria | Bianco | Due pass consecutivi | 75 | 41.0–43.5 | +2.5 pt |
| 7 | Nero | Sconfitta | Bianco | Due pass consecutivi | 130 | 8.0–77.5 | -69.5 pt |
| 8 | Bianco | Vittoria | Bianco | Due pass consecutivi | 71 | 34.0–50.5 | +16.5 pt |
| 9 | Nero | Vittoria | Nero | Due pass consecutivi | 45 | 63.0–24.5 | +38.5 pt |
| 10 | Bianco | Vittoria | Bianco | Due pass consecutivi | 68 | 38.0–50.5 | +12.5 pt |
| 11 | Nero | Vittoria | Nero | Due pass consecutivi | 55 | 45.0–38.5 | +6.5 pt |
| 12 | Bianco | Vittoria | Bianco | Due pass consecutivi | 92 | 43.0–43.5 | +0.5 pt |
| 13 | Nero | Vittoria | Nero | Due pass consecutivi | 46 | 52.0–33.5 | +18.5 pt |
| 14 | Bianco | Vittoria | Bianco | Due pass consecutivi | 65 | 40.0–47.5 | +7.5 pt |
| 15 | Nero | Vittoria | Nero | Due pass consecutivi | 52 | 50.0–31.5 | +18.5 pt |
| 16 | Bianco | Sconfitta | Nero | Due pass consecutivi | 110 | 63.0–19.5 | -43.5 pt |
| 17 | Nero | Vittoria | Nero | Due pass consecutivi | 64 | 46.0–42.5 | +3.5 pt |
| 18 | Bianco | Vittoria | Bianco | Due pass consecutivi | 73 | 36.0–47.5 | +11.5 pt |
| 19 | Nero | Sconfitta | Bianco | Due pass consecutivi | 144 | 12.0–73.5 | -61.5 pt |
| 20 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 34.0–52.5 | +18.5 pt |
| 21 | Nero | Vittoria | Nero | Due pass consecutivi | 64 | 48.0–36.5 | +11.5 pt |
| 22 | Bianco | Vittoria | Bianco | Due pass consecutivi | 68 | 38.0–48.5 | +10.5 pt |
| 23 | Nero | Vittoria | Nero | Due pass consecutivi | 109 | 45.0–40.5 | +4.5 pt |
| 24 | Bianco | Vittoria | Bianco | Due pass consecutivi | 100 | 43.0–45.5 | +2.5 pt |
| 25 | Nero | Vittoria | Nero | Due pass consecutivi | 63 | 50.0–38.5 | +11.5 pt |
| 26 | Bianco | Vittoria | Bianco | Due pass consecutivi | 55 | 36.0–50.5 | +14.5 pt |
| 27 | Nero | Sconfitta | Bianco | Due pass consecutivi | 144 | 10.0–76.5 | -66.5 pt |
| 28 | Bianco | Vittoria | Bianco | Due pass consecutivi | 58 | 28.0–59.5 | +31.5 pt |
| 29 | Nero | Vittoria | Nero | Due pass consecutivi | 59 | 45.0–40.5 | +4.5 pt |
| 30 | Bianco | Sconfitta | Nero | Due pass consecutivi | 131 | 73.0–12.5 | -60.5 pt |
| 31 | Nero | Sconfitta | Bianco | Due pass consecutivi | 164 | 2.0–84.5 | -82.5 pt |
| 32 | Bianco | Vittoria | Bianco | Due pass consecutivi | 79 | 39.0–42.5 | +3.5 pt |
| 33 | Nero | Sconfitta | Bianco | Due pass consecutivi | 156 | 3.0–83.5 | -80.5 pt |
| 34 | Bianco | Sconfitta | Nero | Due pass consecutivi | 132 | 61.0–21.5 | -39.5 pt |
| 35 | Nero | Sconfitta | Bianco | Due pass consecutivi | 140 | 6.0–78.5 | -72.5 pt |
| 36 | Bianco | Vittoria | Bianco | Due pass consecutivi | 71 | 40.0–45.5 | +5.5 pt |
| 37 | Nero | Vittoria | Nero | Due pass consecutivi | 73 | 43.0–40.5 | +2.5 pt |
| 38 | Bianco | Vittoria | Bianco | Due pass consecutivi | 74 | 39.0–40.5 | +1.5 pt |
| 39 | Nero | Vittoria | Nero | Due pass consecutivi | 56 | 47.0–37.5 | +9.5 pt |
| 40 | Bianco | Sconfitta | Nero | Due pass consecutivi | 109 | 57.0–29.5 | -27.5 pt |
| 41 | Nero | Vittoria | Nero | Due pass consecutivi | 107 | 45.0–39.5 | +5.5 pt |
| 42 | Bianco | Vittoria | Bianco | Due pass consecutivi | 71 | 40.0–43.5 | +3.5 pt |
| 43 | Nero | Vittoria | Nero | Due pass consecutivi | 76 | 51.0–36.5 | +14.5 pt |
| 44 | Bianco | Vittoria | Bianco | Due pass consecutivi | 90 | 41.0–41.5 | +0.5 pt |
| 45 | Nero | Sconfitta | Bianco | Due pass consecutivi | 140 | 0.0–88.5 | -88.5 pt |
| 46 | Bianco | Vittoria | Bianco | Due pass consecutivi | 102 | 43.0–43.5 | +0.5 pt |
| 47 | Nero | Vittoria | Nero | Due pass consecutivi | 51 | 52.0–32.5 | +19.5 pt |
| 48 | Bianco | Sconfitta | Nero | Due pass consecutivi | 104 | 59.0–25.5 | -33.5 pt |
| 49 | Nero | Sconfitta | Bianco | Due pass consecutivi | 128 | 8.0–76.5 | -68.5 pt |
| 50 | Bianco | Vittoria | Bianco | Due pass consecutivi | 70 | 40.0–48.5 | +8.5 pt |
| 51 | Nero | Sconfitta | Bianco | Due pass consecutivi | 130 | 12.0–70.5 | -58.5 pt |
| 52 | Bianco | Sconfitta | Nero | Due pass consecutivi | 140 | 69.0–17.5 | -51.5 pt |
| 53 | Nero | Sconfitta | Bianco | Due pass consecutivi | 142 | 0.0–88.5 | -88.5 pt |
| 54 | Bianco | Sconfitta | Nero | Due pass consecutivi | 153 | 72.0–12.5 | -59.5 pt |
| 55 | Nero | Vittoria | Nero | Due pass consecutivi | 73 | 43.0–42.5 | +0.5 pt |
| 56 | Bianco | Vittoria | Bianco | Due pass consecutivi | 64 | 34.0–52.5 | +18.5 pt |
| 57 | Nero | Vittoria | Nero | Due pass consecutivi | 63 | 44.0–42.5 | +1.5 pt |
| 58 | Bianco | Vittoria | Bianco | Due pass consecutivi | 90 | 42.0–43.5 | +1.5 pt |
| 59 | Nero | Sconfitta | Bianco | Due pass consecutivi | 164 | 3.0–82.5 | -79.5 pt |
| 60 | Bianco | Vittoria | Bianco | Due pass consecutivi | 62 | 40.0–47.5 | +7.5 pt |
| 61 | Nero | Vittoria | Nero | Due pass consecutivi | 93 | 47.0–41.5 | +5.5 pt |
| 62 | Bianco | Vittoria | Bianco | Due pass consecutivi | 68 | 38.0–50.5 | +12.5 pt |
| 63 | Nero | Vittoria | Nero | Due pass consecutivi | 46 | 49.0–34.5 | +14.5 pt |
| 64 | Bianco | Vittoria | Bianco | Due pass consecutivi | 78 | 40.0–42.5 | +2.5 pt |
| 65 | Nero | Vittoria | Nero | Due pass consecutivi | 73 | 42.0–41.5 | +0.5 pt |
| 66 | Bianco | Vittoria | Bianco | Due pass consecutivi | 75 | 36.0–48.5 | +12.5 pt |
| 67 | Nero | Vittoria | Nero | Due pass consecutivi | 56 | 43.0–39.5 | +3.5 pt |
| 68 | Bianco | Vittoria | Bianco | Due pass consecutivi | 90 | 40.0–43.5 | +3.5 pt |
| 69 | Nero | Sconfitta | Bianco | Due pass consecutivi | 168 | 4.0–80.5 | -76.5 pt |
| 70 | Bianco | Vittoria | Bianco | Due pass consecutivi | 71 | 40.0–46.5 | +6.5 pt |
| 71 | Nero | Sconfitta | Bianco | Due pass consecutivi | 128 | 3.0–82.5 | -79.5 pt |
| 72 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 32.0–53.5 | +21.5 pt |
| 73 | Nero | Sconfitta | Bianco | Due pass consecutivi | 148 | 2.0–85.5 | -83.5 pt |
| 74 | Bianco | Vittoria | Bianco | Due pass consecutivi | 57 | 44.0–44.5 | +0.5 pt |
| 75 | Nero | Sconfitta | Bianco | Due pass consecutivi | 134 | 3.0–82.5 | -79.5 pt |
| 76 | Bianco | Sconfitta | Nero | Due pass consecutivi | 124 | 60.0–22.5 | -37.5 pt |
| 77 | Nero | Sconfitta | Bianco | Due pass consecutivi | 130 | 7.0–77.5 | -70.5 pt |
| 78 | Bianco | Vittoria | Bianco | Due pass consecutivi | 73 | 37.0–47.5 | +10.5 pt |
| 79 | Nero | Vittoria | Nero | Due pass consecutivi | 63 | 45.0–40.5 | +4.5 pt |
| 80 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 36.0–51.5 | +15.5 pt |
| 81 | Nero | Sconfitta | Bianco | Due pass consecutivi | 133 | 8.0–74.5 | -66.5 pt |
| 82 | Bianco | Vittoria | Bianco | Due pass consecutivi | 76 | 43.0–44.5 | +1.5 pt |
| 83 | Nero | Sconfitta | Bianco | Due pass consecutivi | 156 | 5.0–80.5 | -75.5 pt |
| 84 | Bianco | Vittoria | Bianco | Due pass consecutivi | 60 | 37.0–51.5 | +14.5 pt |
| 85 | Nero | Sconfitta | Bianco | Due pass consecutivi | 150 | 10.0–71.5 | -61.5 pt |
| 86 | Bianco | Vittoria | Bianco | Due pass consecutivi | 58 | 32.0–47.5 | +15.5 pt |
| 87 | Nero | Sconfitta | Bianco | Due pass consecutivi | 138 | 4.0–81.5 | -77.5 pt |
| 88 | Bianco | Sconfitta | Nero | Due pass consecutivi | 124 | 58.0–25.5 | -32.5 pt |
| 89 | Nero | Sconfitta | Bianco | Due pass consecutivi | 134 | 8.0–76.5 | -68.5 pt |
| 90 | Bianco | Vittoria | Bianco | Due pass consecutivi | 66 | 34.0–47.5 | +13.5 pt |
| 91 | Nero | Sconfitta | Bianco | Due pass consecutivi | 126 | 7.0–77.5 | -70.5 pt |
| 92 | Bianco | Sconfitta | Nero | Due pass consecutivi | 153 | 79.0–8.5 | -70.5 pt |
| 93 | Nero | Vittoria | Nero | Due pass consecutivi | 89 | 43.0–40.5 | +2.5 pt |
| 94 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 37.0–43.5 | +6.5 pt |
| 95 | Nero | Vittoria | Nero | Due pass consecutivi | 76 | 43.0–39.5 | +3.5 pt |
| 96 | Bianco | Vittoria | Bianco | Due pass consecutivi | 63 | 35.0–45.5 | +10.5 pt |
| 97 | Nero | Sconfitta | Bianco | Due pass consecutivi | 136 | 5.0–80.5 | -75.5 pt |
| 98 | Bianco | Vittoria | Bianco | Due pass consecutivi | 64 | 37.0–51.5 | +14.5 pt |
| 99 | Nero | Vittoria | Nero | Due pass consecutivi | 73 | 43.0–40.5 | +2.5 pt |
| 100 | Bianco | Vittoria | Bianco | Due pass consecutivi | 68 | 38.0–45.5 | +7.5 pt |

## Altri file

- CSV dettagliato: `TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.csv`
- Full report JSON: `TRM/model_evaluation/results/9x9_go/full_reports/negamax-d2-c5-rank20k-40v-100g.report.json`

