# Riepilogo test TRM vs KataGo — 9x9

- **Stato:** COMPLETATO
- **Inizio (UTC):** 2026-11-13 20:02:23
- **Fine (UTC):** 2026-11-13 20:05:04
- **Durata:** 2.7 minuti (160.7 secondi)
- **Avversario:** KataGo 1.18.2+kata9x9-b18c384nbt-s6603M+b18c384nbt-humanv0
- **Modalità TRM:** policy

## Parametri del test

Comando eseguito:
```bash
python3 model_evaluation/go_match.py --board-size 9 --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago --katago-config model_evaluation/katago-local/engine/default_gtp.cfg --katago-model model_evaluation/katago-local/downloads/kata9x9-b18c384nbt-20231025.bin.gz --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz --human-profile rank_20k --games 100 --max-visits 40 --seed 42 --pass-logit-penalty 3.5 --pass-penalty-before-stones 16 --search policy --output model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.csv
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
| --search | policy |
| --search-depth | 2 |
| --search-candidates | — |
| --max-visits | 40 |
| --gtp-timeout | 600.0 |
| --device | auto |
| --pass-logit-penalty | 3.5 |
| --pass-penalty-before-stones | 16 |
| --seed | 42 |
| --output | TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.csv |
| --report-output | TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.report.json |
| --summary-output | TRM/model_evaluation/results/9x9_go/summaries/policy-rank20k-40v-100g.summary.md |

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
| Profondità Negamax effettiva | 0 |
| CSV partite | TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.csv |
| Full report JSON | TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.report.json |
| Riepilogo Markdown | TRM/model_evaluation/results/9x9_go/summaries/policy-rank20k-40v-100g.summary.md |

## Risultati complessivi

| Indicatore | Risultato |
| --- | --- |
| Partite richieste | 100 |
| Partite registrate | 100 |
| Partite completate | 100 |
| Partite incomplete | 0 |
| Partite non iniziate | 0 |
| Vittorie TRM | 26 |
| Pareggi | 0 |
| Sconfitte TRM (vittorie KataGo) | 74 |
| Win rate TRM | 26.0% |
| Score rate TRM (vittoria=1, pareggio=0.5) | 26.0% |
| Intervallo 95% win rate (Wilson) | 18.4%–35.4% |
| Gap medio TRM | -7.3 pt |
| Gap mediano TRM | -7.5 pt |
| Gap medio assoluto | 14.6 pt |
| Partite con punteggio numerico | 100 |
| Gap a favore TRM / a favore KataGo / pari | 26 / 74 / 0 |
| Mosse medie per partita registrata | 63.9 |
| PASS giocati dal TRM | 169 |
| PASS prima di 16 pietre | 0 |
| Decisioni con penalità PASS attiva | 809 |
| Terminazioni | two_passes: 100 |

*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*

## Risultati per colore del TRM

| Colore TRM | Giocate | Completate | V-P-S | Win rate | Gap medio |
| --- | --- | --- | --- | --- | --- |
| Nero | 50 | 50 | 7-0-43 | 14.0% | -11.4 pt |
| Bianco | 50 | 50 | 19-0-31 | 38.0% | -3.2 pt |

## Statistiche di ricerca

| Indicatore | Totale |
| --- | --- |
| Nodi esplorati | 0 |
| Foglie valutate | 0 |
| Cutoff | 0 |
| Chiamate alla policy | 0 |
| Tempo totale ricerca | 0.00 s |

*Le statistiche di ricerca sono specifiche del Negamax; con `--search policy` restano a zero.*

## Risultato di ogni partita

| # | Colore TRM | Esito TRM | Vincitore | Terminazione | Mosse | Score B–W | Gap TRM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Nero | Sconfitta | Bianco | Due pass consecutivi | 83 | 35.0–50.5 | -15.5 pt |
| 2 | Bianco | Vittoria | Bianco | Due pass consecutivi | 49 | 33.0–44.5 | +11.5 pt |
| 3 | Nero | Sconfitta | Bianco | Due pass consecutivi | 65 | 41.0–43.5 | -2.5 pt |
| 4 | Bianco | Vittoria | Bianco | Due pass consecutivi | 50 | 29.0–59.5 | +30.5 pt |
| 5 | Nero | Sconfitta | Bianco | Due pass consecutivi | 49 | 30.0–42.5 | -12.5 pt |
| 6 | Bianco | Sconfitta | Nero | Due pass consecutivi | 49 | 46.0–41.5 | -4.5 pt |
| 7 | Nero | Sconfitta | Bianco | Due pass consecutivi | 81 | 27.0–59.5 | -32.5 pt |
| 8 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 50.0–38.5 | -11.5 pt |
| 9 | Nero | Sconfitta | Bianco | Due pass consecutivi | 52 | 35.0–53.5 | -18.5 pt |
| 10 | Bianco | Vittoria | Bianco | Due pass consecutivi | 53 | 34.0–54.5 | +20.5 pt |
| 11 | Nero | Sconfitta | Bianco | Due pass consecutivi | 78 | 33.0–51.5 | -18.5 pt |
| 12 | Bianco | Sconfitta | Nero | Due pass consecutivi | 58 | 42.0–27.5 | -14.5 pt |
| 13 | Nero | Sconfitta | Bianco | Due pass consecutivi | 69 | 44.0–44.5 | -0.5 pt |
| 14 | Bianco | Sconfitta | Nero | Due pass consecutivi | 66 | 52.0–36.5 | -15.5 pt |
| 15 | Nero | Sconfitta | Bianco | Due pass consecutivi | 56 | 12.0–48.5 | -36.5 pt |
| 16 | Bianco | Sconfitta | Nero | Due pass consecutivi | 77 | 41.0–40.5 | -0.5 pt |
| 17 | Nero | Sconfitta | Bianco | Due pass consecutivi | 74 | 36.0–51.5 | -15.5 pt |
| 18 | Bianco | Sconfitta | Nero | Due pass consecutivi | 51 | 48.0–29.5 | -18.5 pt |
| 19 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 41.0–45.5 | -4.5 pt |
| 20 | Bianco | Vittoria | Bianco | Due pass consecutivi | 56 | 38.0–40.5 | +2.5 pt |
| 21 | Nero | Sconfitta | Bianco | Due pass consecutivi | 65 | 30.0–41.5 | -11.5 pt |
| 22 | Bianco | Vittoria | Bianco | Due pass consecutivi | 53 | 36.0–52.5 | +16.5 pt |
| 23 | Nero | Sconfitta | Bianco | Due pass consecutivi | 56 | 30.0–50.5 | -20.5 pt |
| 24 | Bianco | Sconfitta | Nero | Due pass consecutivi | 63 | 48.0–40.5 | -7.5 pt |
| 25 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 17.0–48.5 | -31.5 pt |
| 26 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 44.0–44.5 | +0.5 pt |
| 27 | Nero | Sconfitta | Bianco | Due pass consecutivi | 60 | 35.0–53.5 | -18.5 pt |
| 28 | Bianco | Vittoria | Bianco | Due pass consecutivi | 54 | 32.0–35.5 | +3.5 pt |
| 29 | Nero | Sconfitta | Bianco | Due pass consecutivi | 83 | 40.0–46.5 | -6.5 pt |
| 30 | Bianco | Sconfitta | Nero | Due pass consecutivi | 73 | 55.0–33.5 | -21.5 pt |
| 31 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 41.0–45.5 | -4.5 pt |
| 32 | Bianco | Sconfitta | Nero | Due pass consecutivi | 73 | 44.0–42.5 | -1.5 pt |
| 33 | Nero | Sconfitta | Bianco | Due pass consecutivi | 60 | 42.0–46.5 | -4.5 pt |
| 34 | Bianco | Vittoria | Bianco | Due pass consecutivi | 61 | 42.0–43.5 | +1.5 pt |
| 35 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 41.0–45.5 | -4.5 pt |
| 36 | Bianco | Vittoria | Bianco | Due pass consecutivi | 62 | 31.0–44.5 | +13.5 pt |
| 37 | Nero | Vittoria | Nero | Due pass consecutivi | 54 | 54.0–34.5 | +19.5 pt |
| 38 | Bianco | Sconfitta | Nero | Due pass consecutivi | 59 | 41.0–36.5 | -4.5 pt |
| 39 | Nero | Sconfitta | Bianco | Due pass consecutivi | 84 | 19.0–65.5 | -46.5 pt |
| 40 | Bianco | Vittoria | Bianco | Due pass consecutivi | 61 | 35.0–52.5 | +17.5 pt |
| 41 | Nero | Vittoria | Nero | Due pass consecutivi | 77 | 41.0–31.5 | +9.5 pt |
| 42 | Bianco | Sconfitta | Nero | Due pass consecutivi | 63 | 48.0–40.5 | -7.5 pt |
| 43 | Nero | Sconfitta | Bianco | Due pass consecutivi | 56 | 44.0–44.5 | -0.5 pt |
| 44 | Bianco | Sconfitta | Nero | Due pass consecutivi | 55 | 50.0–38.5 | -11.5 pt |
| 45 | Nero | Vittoria | Nero | Due pass consecutivi | 57 | 33.0–23.5 | +9.5 pt |
| 46 | Bianco | Sconfitta | Nero | Due pass consecutivi | 74 | 47.0–40.5 | -6.5 pt |
| 47 | Nero | Sconfitta | Bianco | Due pass consecutivi | 75 | 42.0–45.5 | -3.5 pt |
| 48 | Bianco | Sconfitta | Nero | Due pass consecutivi | 63 | 68.0–20.5 | -47.5 pt |
| 49 | Nero | Sconfitta | Bianco | Due pass consecutivi | 62 | 24.0–41.5 | -17.5 pt |
| 50 | Bianco | Sconfitta | Nero | Due pass consecutivi | 85 | 44.0–42.5 | -1.5 pt |
| 51 | Nero | Sconfitta | Bianco | Due pass consecutivi | 80 | 30.0–55.5 | -25.5 pt |
| 52 | Bianco | Vittoria | Bianco | Due pass consecutivi | 35 | 37.0–51.5 | +14.5 pt |
| 53 | Nero | Sconfitta | Bianco | Due pass consecutivi | 75 | 38.0–49.5 | -11.5 pt |
| 54 | Bianco | Sconfitta | Nero | Due pass consecutivi | 73 | 68.0–16.5 | -51.5 pt |
| 55 | Nero | Sconfitta | Bianco | Due pass consecutivi | 82 | 34.0–52.5 | -18.5 pt |
| 56 | Bianco | Sconfitta | Nero | Due pass consecutivi | 53 | 45.0–43.5 | -1.5 pt |
| 57 | Nero | Vittoria | Nero | Due pass consecutivi | 77 | 58.0–28.5 | +29.5 pt |
| 58 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 52.0–35.5 | -16.5 pt |
| 59 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 38.0–48.5 | -10.5 pt |
| 60 | Bianco | Vittoria | Bianco | Due pass consecutivi | 64 | 42.0–44.5 | +2.5 pt |
| 61 | Nero | Sconfitta | Bianco | Due pass consecutivi | 45 | 38.0–50.5 | -12.5 pt |
| 62 | Bianco | Vittoria | Bianco | Due pass consecutivi | 48 | 40.0–48.5 | +8.5 pt |
| 63 | Nero | Sconfitta | Bianco | Due pass consecutivi | 66 | 32.0–54.5 | -22.5 pt |
| 64 | Bianco | Vittoria | Bianco | Due pass consecutivi | 44 | 28.0–59.5 | +31.5 pt |
| 65 | Nero | Sconfitta | Bianco | Due pass consecutivi | 58 | 32.0–45.5 | -13.5 pt |
| 66 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 50.0–36.5 | -13.5 pt |
| 67 | Nero | Vittoria | Nero | Due pass consecutivi | 57 | 33.0–23.5 | +9.5 pt |
| 68 | Bianco | Vittoria | Bianco | Due pass consecutivi | 39 | 31.0–57.5 | +26.5 pt |
| 69 | Nero | Sconfitta | Bianco | Due pass consecutivi | 80 | 18.0–63.5 | -45.5 pt |
| 70 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 55.0–33.5 | -21.5 pt |
| 71 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 41.0–45.5 | -4.5 pt |
| 72 | Bianco | Sconfitta | Nero | Due pass consecutivi | 50 | 36.0–26.5 | -9.5 pt |
| 73 | Nero | Sconfitta | Bianco | Due pass consecutivi | 55 | 28.0–35.5 | -7.5 pt |
| 74 | Bianco | Sconfitta | Nero | Due pass consecutivi | 64 | 45.0–43.5 | -1.5 pt |
| 75 | Nero | Sconfitta | Bianco | Due pass consecutivi | 68 | 38.0–42.5 | -4.5 pt |
| 76 | Bianco | Sconfitta | Nero | Due pass consecutivi | 74 | 45.0–41.5 | -3.5 pt |
| 77 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 14.0–74.5 | -60.5 pt |
| 78 | Bianco | Sconfitta | Nero | Due pass consecutivi | 83 | 52.0–33.5 | -18.5 pt |
| 79 | Nero | Sconfitta | Bianco | Due pass consecutivi | 44 | 41.0–47.5 | -6.5 pt |
| 80 | Bianco | Sconfitta | Nero | Due pass consecutivi | 65 | 42.0–32.5 | -9.5 pt |
| 81 | Nero | Vittoria | Nero | Due pass consecutivi | 57 | 33.0–23.5 | +9.5 pt |
| 82 | Bianco | Vittoria | Bianco | Due pass consecutivi | 80 | 36.0–50.5 | +14.5 pt |
| 83 | Nero | Sconfitta | Bianco | Due pass consecutivi | 62 | 37.0–46.5 | -9.5 pt |
| 84 | Bianco | Sconfitta | Nero | Due pass consecutivi | 63 | 48.0–40.5 | -7.5 pt |
| 85 | Nero | Vittoria | Nero | Due pass consecutivi | 62 | 58.0–30.5 | +27.5 pt |
| 86 | Bianco | Vittoria | Bianco | Due pass consecutivi | 29 | 34.0–54.5 | +20.5 pt |
| 87 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 41.0–45.5 | -4.5 pt |
| 88 | Bianco | Vittoria | Bianco | Due pass consecutivi | 34 | 43.0–45.5 | +2.5 pt |
| 89 | Nero | Sconfitta | Bianco | Due pass consecutivi | 82 | 31.0–56.5 | -25.5 pt |
| 90 | Bianco | Vittoria | Bianco | Due pass consecutivi | 35 | 37.0–51.5 | +14.5 pt |
| 91 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 36.0–51.5 | -15.5 pt |
| 92 | Bianco | Sconfitta | Nero | Due pass consecutivi | 71 | 46.0–35.5 | -10.5 pt |
| 93 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 41.0–45.5 | -4.5 pt |
| 94 | Bianco | Sconfitta | Nero | Due pass consecutivi | 75 | 48.0–40.5 | -7.5 pt |
| 95 | Nero | Sconfitta | Bianco | Due pass consecutivi | 37 | 18.0–39.5 | -21.5 pt |
| 96 | Bianco | Sconfitta | Nero | Due pass consecutivi | 48 | 35.0–25.5 | -9.5 pt |
| 97 | Nero | Sconfitta | Bianco | Due pass consecutivi | 67 | 37.0–49.5 | -12.5 pt |
| 98 | Bianco | Sconfitta | Nero | Due pass consecutivi | 52 | 49.0–39.5 | -9.5 pt |
| 99 | Nero | Sconfitta | Bianco | Due pass consecutivi | 82 | 34.0–52.5 | -18.5 pt |
| 100 | Bianco | Sconfitta | Nero | Due pass consecutivi | 81 | 62.0–16.5 | -45.5 pt |

## Altri file

- CSV dettagliato: `TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.csv`
- Full report JSON: `TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.report.json`

