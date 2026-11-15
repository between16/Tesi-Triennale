# Riepilogo test TRM vs KataGo — 9x9

- **Stato:** COMPLETATO
- **Inizio (UTC):** 2026-11-15 13:36:47
- **Fine (UTC):** 2026-11-15 13:39:28
- **Durata:** 2.7 minuti (160.5 secondi)
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
| Gap medio TRM | -13.4 pt |
| Gap mediano TRM | -11.5 pt |
| Gap medio assoluto | 21.2 pt |
| Partite con punteggio numerico | 100 |
| Gap a favore TRM / a favore KataGo / pari | 26 / 74 / 0 |
| Mosse medie per partita registrata | 64.0 |
| PASS giocati dal TRM | 190 |
| PASS prima di 16 pietre | 0 |
| Decisioni con penalità PASS attiva | 818 |
| Terminazioni | two_passes: 100 |

*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*

## Risultati per colore del TRM

| Colore TRM | Giocate | Completate | V-P-S | Win rate | Gap medio |
| --- | --- | --- | --- | --- | --- |
| Nero | 50 | 50 | 7-0-43 | 14.0% | -20.1 pt |
| Bianco | 50 | 50 | 19-0-31 | 38.0% | -6.7 pt |

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
| 1 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 34.0–48.5 | -14.5 pt |
| 2 | Bianco | Sconfitta | Nero | Due pass consecutivi | 65 | 43.0–38.5 | -4.5 pt |
| 3 | Nero | Sconfitta | Bianco | Due pass consecutivi | 52 | 37.0–51.5 | -14.5 pt |
| 4 | Bianco | Sconfitta | Nero | Due pass consecutivi | 57 | 49.0–38.5 | -10.5 pt |
| 5 | Nero | Sconfitta | Bianco | Due pass consecutivi | 64 | 42.0–44.5 | -2.5 pt |
| 6 | Bianco | Sconfitta | Nero | Due pass consecutivi | 60 | 45.0–43.5 | -1.5 pt |
| 7 | Nero | Sconfitta | Bianco | Due pass consecutivi | 41 | 18.0–36.5 | -18.5 pt |
| 8 | Bianco | Sconfitta | Nero | Due pass consecutivi | 47 | 43.0–42.5 | -0.5 pt |
| 9 | Nero | Sconfitta | Bianco | Due pass consecutivi | 80 | 24.0–62.5 | -38.5 pt |
| 10 | Bianco | Vittoria | Bianco | Due pass consecutivi | 57 | 33.0–55.5 | +22.5 pt |
| 11 | Nero | Sconfitta | Bianco | Due pass consecutivi | 78 | 31.0–54.5 | -23.5 pt |
| 12 | Bianco | Sconfitta | Nero | Due pass consecutivi | 85 | 69.0–19.5 | -49.5 pt |
| 13 | Nero | Vittoria | Nero | Due pass consecutivi | 57 | 49.0–34.5 | +14.5 pt |
| 14 | Bianco | Sconfitta | Nero | Due pass consecutivi | 55 | 46.0–42.5 | -3.5 pt |
| 15 | Nero | Sconfitta | Bianco | Due pass consecutivi | 74 | 34.0–48.5 | -14.5 pt |
| 16 | Bianco | Sconfitta | Nero | Due pass consecutivi | 69 | 64.0–12.5 | -51.5 pt |
| 17 | Nero | Sconfitta | Bianco | Due pass consecutivi | 55 | 44.0–44.5 | -0.5 pt |
| 18 | Bianco | Sconfitta | Nero | Due pass consecutivi | 61 | 57.0–31.5 | -25.5 pt |
| 19 | Nero | Sconfitta | Bianco | Due pass consecutivi | 70 | 17.0–70.5 | -53.5 pt |
| 20 | Bianco | Vittoria | Bianco | Due pass consecutivi | 79 | 43.0–43.5 | +0.5 pt |
| 21 | Nero | Sconfitta | Bianco | Due pass consecutivi | 47 | 44.0–44.5 | -0.5 pt |
| 22 | Bianco | Sconfitta | Nero | Due pass consecutivi | 77 | 66.0–14.5 | -51.5 pt |
| 23 | Nero | Sconfitta | Bianco | Due pass consecutivi | 82 | 18.0–57.5 | -39.5 pt |
| 24 | Bianco | Sconfitta | Nero | Due pass consecutivi | 54 | 50.0–38.5 | -11.5 pt |
| 25 | Nero | Sconfitta | Bianco | Due pass consecutivi | 50 | 43.0–43.5 | -0.5 pt |
| 26 | Bianco | Sconfitta | Nero | Due pass consecutivi | 70 | 57.0–24.5 | -32.5 pt |
| 27 | Nero | Vittoria | Nero | Due pass consecutivi | 45 | 47.0–41.5 | +5.5 pt |
| 28 | Bianco | Vittoria | Bianco | Due pass consecutivi | 48 | 29.0–59.5 | +30.5 pt |
| 29 | Nero | Sconfitta | Bianco | Due pass consecutivi | 43 | 24.0–32.5 | -8.5 pt |
| 30 | Bianco | Vittoria | Bianco | Due pass consecutivi | 54 | 23.0–65.5 | +42.5 pt |
| 31 | Nero | Sconfitta | Bianco | Due pass consecutivi | 68 | 33.0–48.5 | -15.5 pt |
| 32 | Bianco | Vittoria | Bianco | Due pass consecutivi | 49 | 41.0–47.5 | +6.5 pt |
| 33 | Nero | Sconfitta | Bianco | Due pass consecutivi | 56 | 35.0–46.5 | -11.5 pt |
| 34 | Bianco | Sconfitta | Nero | Due pass consecutivi | 63 | 49.0–37.5 | -11.5 pt |
| 35 | Nero | Sconfitta | Bianco | Due pass consecutivi | 81 | 34.0–48.5 | -14.5 pt |
| 36 | Bianco | Vittoria | Bianco | Due pass consecutivi | 43 | 30.0–41.5 | +11.5 pt |
| 37 | Nero | Sconfitta | Bianco | Due pass consecutivi | 43 | 25.0–41.5 | -16.5 pt |
| 38 | Bianco | Vittoria | Bianco | Due pass consecutivi | 48 | 29.0–59.5 | +30.5 pt |
| 39 | Nero | Sconfitta | Bianco | Due pass consecutivi | 63 | 34.0–39.5 | -5.5 pt |
| 40 | Bianco | Sconfitta | Nero | Due pass consecutivi | 66 | 55.0–33.5 | -21.5 pt |
| 41 | Nero | Sconfitta | Bianco | Due pass consecutivi | 58 | 39.0–49.5 | -10.5 pt |
| 42 | Bianco | Sconfitta | Nero | Due pass consecutivi | 61 | 39.0–24.5 | -14.5 pt |
| 43 | Nero | Vittoria | Nero | Due pass consecutivi | 57 | 53.0–34.5 | +18.5 pt |
| 44 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 49.0–38.5 | -10.5 pt |
| 45 | Nero | Sconfitta | Bianco | Due pass consecutivi | 55 | 44.0–44.5 | -0.5 pt |
| 46 | Bianco | Sconfitta | Nero | Due pass consecutivi | 81 | 44.0–42.5 | -1.5 pt |
| 47 | Nero | Sconfitta | Bianco | Due pass consecutivi | 84 | 31.0–54.5 | -23.5 pt |
| 48 | Bianco | Vittoria | Bianco | Due pass consecutivi | 50 | 22.0–39.5 | +17.5 pt |
| 49 | Nero | Sconfitta | Bianco | Due pass consecutivi | 84 | 32.0–55.5 | -23.5 pt |
| 50 | Bianco | Vittoria | Bianco | Due pass consecutivi | 39 | 30.0–39.5 | +9.5 pt |
| 51 | Nero | Vittoria | Nero | Due pass consecutivi | 38 | 55.0–32.5 | +22.5 pt |
| 52 | Bianco | Sconfitta | Nero | Due pass consecutivi | 77 | 56.0–23.5 | -32.5 pt |
| 53 | Nero | Vittoria | Nero | Due pass consecutivi | 91 | 51.0–37.5 | +13.5 pt |
| 54 | Bianco | Sconfitta | Nero | Due pass consecutivi | 85 | 58.0–25.5 | -32.5 pt |
| 55 | Nero | Sconfitta | Bianco | Due pass consecutivi | 76 | 2.0–80.5 | -78.5 pt |
| 56 | Bianco | Vittoria | Bianco | Due pass consecutivi | 60 | 35.0–45.5 | +10.5 pt |
| 57 | Nero | Sconfitta | Bianco | Due pass consecutivi | 84 | 29.0–59.5 | -30.5 pt |
| 58 | Bianco | Sconfitta | Nero | Due pass consecutivi | 64 | 44.0–24.5 | -19.5 pt |
| 59 | Nero | Sconfitta | Bianco | Due pass consecutivi | 50 | 19.0–44.5 | -25.5 pt |
| 60 | Bianco | Vittoria | Bianco | Due pass consecutivi | 61 | 39.0–48.5 | +9.5 pt |
| 61 | Nero | Sconfitta | Bianco | Due pass consecutivi | 44 | 18.0–38.5 | -20.5 pt |
| 62 | Bianco | Vittoria | Bianco | Due pass consecutivi | 67 | 44.0–44.5 | +0.5 pt |
| 63 | Nero | Vittoria | Nero | Due pass consecutivi | 45 | 40.0–35.5 | +4.5 pt |
| 64 | Bianco | Vittoria | Bianco | Due pass consecutivi | 39 | 36.0–52.5 | +16.5 pt |
| 65 | Nero | Sconfitta | Bianco | Due pass consecutivi | 70 | 11.0–59.5 | -48.5 pt |
| 66 | Bianco | Sconfitta | Nero | Due pass consecutivi | 63 | 49.0–38.5 | -10.5 pt |
| 67 | Nero | Sconfitta | Bianco | Due pass consecutivi | 70 | 17.0–70.5 | -53.5 pt |
| 68 | Bianco | Sconfitta | Nero | Due pass consecutivi | 91 | 56.0–30.5 | -25.5 pt |
| 69 | Nero | Sconfitta | Bianco | Due pass consecutivi | 86 | 0.0–88.5 | -88.5 pt |
| 70 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 49.0–39.5 | -9.5 pt |
| 71 | Nero | Sconfitta | Bianco | Due pass consecutivi | 60 | 32.0–49.5 | -17.5 pt |
| 72 | Bianco | Sconfitta | Nero | Due pass consecutivi | 57 | 46.0–42.5 | -3.5 pt |
| 73 | Nero | Sconfitta | Bianco | Due pass consecutivi | 59 | 43.0–45.5 | -2.5 pt |
| 74 | Bianco | Sconfitta | Nero | Due pass consecutivi | 54 | 59.0–29.5 | -29.5 pt |
| 75 | Nero | Sconfitta | Bianco | Due pass consecutivi | 84 | 29.0–55.5 | -26.5 pt |
| 76 | Bianco | Vittoria | Bianco | Due pass consecutivi | 54 | 29.0–58.5 | +29.5 pt |
| 77 | Nero | Sconfitta | Bianco | Due pass consecutivi | 88 | 2.0–84.5 | -82.5 pt |
| 78 | Bianco | Sconfitta | Nero | Due pass consecutivi | 79 | 51.0–35.5 | -15.5 pt |
| 79 | Nero | Sconfitta | Bianco | Due pass consecutivi | 78 | 33.0–54.5 | -21.5 pt |
| 80 | Bianco | Vittoria | Bianco | Due pass consecutivi | 53 | 38.0–50.5 | +12.5 pt |
| 81 | Nero | Sconfitta | Bianco | Due pass consecutivi | 88 | 0.0–88.5 | -88.5 pt |
| 82 | Bianco | Vittoria | Bianco | Due pass consecutivi | 89 | 41.0–44.5 | +3.5 pt |
| 83 | Nero | Sconfitta | Bianco | Due pass consecutivi | 70 | 30.0–49.5 | -19.5 pt |
| 84 | Bianco | Sconfitta | Nero | Due pass consecutivi | 85 | 69.0–19.5 | -49.5 pt |
| 85 | Nero | Vittoria | Nero | Due pass consecutivi | 48 | 46.0–36.5 | +9.5 pt |
| 86 | Bianco | Sconfitta | Nero | Due pass consecutivi | 59 | 56.0–21.5 | -34.5 pt |
| 87 | Nero | Sconfitta | Bianco | Due pass consecutivi | 74 | 34.0–48.5 | -14.5 pt |
| 88 | Bianco | Sconfitta | Nero | Due pass consecutivi | 85 | 69.0–19.5 | -49.5 pt |
| 89 | Nero | Sconfitta | Bianco | Due pass consecutivi | 54 | 41.0–47.5 | -6.5 pt |
| 90 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 32.0–56.5 | +24.5 pt |
| 91 | Nero | Sconfitta | Bianco | Due pass consecutivi | 48 | 38.0–48.5 | -10.5 pt |
| 92 | Bianco | Vittoria | Bianco | Due pass consecutivi | 59 | 44.0–44.5 | +0.5 pt |
| 93 | Nero | Sconfitta | Bianco | Due pass consecutivi | 79 | 33.0–53.5 | -20.5 pt |
| 94 | Bianco | Sconfitta | Nero | Due pass consecutivi | 70 | 46.0–39.5 | -6.5 pt |
| 95 | Nero | Sconfitta | Bianco | Due pass consecutivi | 66 | 9.0–68.5 | -59.5 pt |
| 96 | Bianco | Sconfitta | Nero | Due pass consecutivi | 60 | 47.0–40.5 | -6.5 pt |
| 97 | Nero | Sconfitta | Bianco | Due pass consecutivi | 43 | 25.0–41.5 | -16.5 pt |
| 98 | Bianco | Sconfitta | Nero | Due pass consecutivi | 67 | 48.0–36.5 | -11.5 pt |
| 99 | Nero | Sconfitta | Bianco | Due pass consecutivi | 83 | 36.0–47.5 | -11.5 pt |
| 100 | Bianco | Vittoria | Bianco | Due pass consecutivi | 53 | 32.0–56.5 | +24.5 pt |

## Altri file

- CSV dettagliato: `TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.csv`
- Full report JSON: `TRM/model_evaluation/results/9x9_go/full_reports/policy-rank20k-40v-100g.report.json`

