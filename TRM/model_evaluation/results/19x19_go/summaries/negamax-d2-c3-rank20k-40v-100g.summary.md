# Riepilogo test TRM vs KataGo — 19x19

- **Stato:** COMPLETATO
- **Inizio (UTC):** 2026-11-13 20:31:54
- **Fine (UTC):** 2026-11-13 21:07:39
- **Durata:** 35.7 minuti (2144.5 secondi)
- **Avversario:** KataGo 1.18.2+tf3-b11c768-s12382M-d6533M+b18c384nbt-humanv0
- **Modalità TRM:** negamax

## Parametri del test

Comando eseguito:
```bash
python3 model_evaluation/go_match.py --board-size 19 --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago --katago-config model_evaluation/katago-local/engine/default_gtp.cfg --katago-model model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz --human-profile rank_20k --games 100 --max-visits 40 --seed 42 --pass-logit-penalty 3.5 --pass-penalty-before-stones 16 --search negamax --search-depth 2 --search-candidates 3 --output model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.csv
```

La tabella riporta gli argomenti CLI (compresi i valori di default) e i parametri effettivi risolti dallo script.

| Argomento CLI | Valore |
| --- | --- |
| --board-size | 19 |
| --checkpoint | TRM/checkpoints/19x19_go/best_model.pt |
| --rules | TRM/model_evaluation/config/rules.json |
| --katago | TRM/model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago |
| --katago-config | TRM/model_evaluation/katago-local/engine/default_gtp.cfg |
| --katago-model | TRM/model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz |
| --katago-human-model | TRM/model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz |
| --human-profile | rank_20k |
| --human-temperature-mode | — |
| --games | 100 |
| --max-moves | — |
| --search | negamax |
| --search-depth | 2 |
| --search-candidates | 3 |
| --max-visits | 40 |
| --gtp-timeout | 600.0 |
| --device | auto |
| --pass-logit-penalty | 3.5 |
| --pass-penalty-before-stones | 16 |
| --seed | 42 |
| --output | TRM/model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.csv |
| --report-output | TRM/model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.report.json |
| --summary-output | TRM/model_evaluation/results/19x19_go/summaries/negamax-d2-c3-rank20k-40v-100g.summary.md |

### Parametri effettivi e file generati

| Parametro | Valore |
| --- | --- |
| Checkpoint TRM | TRM/checkpoints/19x19_go/best_model.pt |
| Device effettivo | cuda |
| Komi | 7.5 |
| Limite mosse effettivo | 1444 |
| Regole GTP effettive | {"friendlyPassOk": true, "hasButton": false, "ko": "POSITIONAL", "scoring": "AREA", "suicide": false, "tax": "NONE", "whiteHandicapBonus": "0"} |
| Profilo HumanSL effettivo | rank_20k |
| Candidati Negamax effettivi | 3 |
| Profondità Negamax effettiva | 2 |
| CSV partite | TRM/model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.csv |
| Full report JSON | TRM/model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.report.json |
| Riepilogo Markdown | TRM/model_evaluation/results/19x19_go/summaries/negamax-d2-c3-rank20k-40v-100g.summary.md |

## Risultati complessivi

| Indicatore | Risultato |
| --- | --- |
| Partite richieste | 100 |
| Partite registrate | 100 |
| Partite completate | 100 |
| Partite incomplete | 0 |
| Partite non iniziate | 0 |
| Vittorie TRM | 65 |
| Pareggi | 0 |
| Sconfitte TRM (vittorie KataGo) | 35 |
| Win rate TRM | 65.0% |
| Score rate TRM (vittoria=1, pareggio=0.5) | 65.0% |
| Intervallo 95% win rate (Wilson) | 55.3%–73.6% |
| Gap medio TRM | -62.8 pt |
| Gap mediano TRM | +3.5 pt |
| Gap medio assoluto | 108.8 pt |
| Partite con punteggio numerico | 100 |
| Gap a favore TRM / a favore KataGo / pari | 65 / 35 / 0 |
| Mosse medie per partita registrata | 393.8 |
| PASS giocati dal TRM | 186 |
| PASS prima di 16 pietre | 0 |
| Decisioni con penalità PASS attiva | 800 |
| Terminazioni | two_passes: 100 |

*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*

## Risultati per colore del TRM

| Colore TRM | Giocate | Completate | V-P-S | Win rate | Gap medio |
| --- | --- | --- | --- | --- | --- |
| Nero | 50 | 50 | 28-0-22 | 56.0% | -88.4 pt |
| Bianco | 50 | 50 | 37-0-13 | 74.0% | -37.3 pt |

## Statistiche di ricerca

| Indicatore | Totale |
| --- | --- |
| Nodi esplorati | 301538 |
| Foglie valutate | 218254 |
| Cutoff | 28886 |
| Chiamate alla policy | 83284 |
| Tempo totale ricerca | 899.38 s |

*Le statistiche di ricerca sono specifiche del Negamax; con `--search policy` restano a zero.*

## Risultato di ogni partita

| # | Colore TRM | Esito TRM | Vincitore | Terminazione | Mosse | Score B–W | Gap TRM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Nero | Sconfitta | Bianco | Due pass consecutivi | 744 | 22.0–341.5 | -319.5 pt |
| 2 | Bianco | Vittoria | Bianco | Due pass consecutivi | 112 | 143.0–225.5 | +82.5 pt |
| 3 | Nero | Vittoria | Nero | Due pass consecutivi | 219 | 195.0–162.5 | +32.5 pt |
| 4 | Bianco | Sconfitta | Nero | Due pass consecutivi | 523 | 317.0–51.5 | -265.5 pt |
| 5 | Nero | Sconfitta | Bianco | Due pass consecutivi | 527 | 57.0–307.5 | -250.5 pt |
| 6 | Bianco | Vittoria | Bianco | Due pass consecutivi | 291 | 151.0–180.5 | +29.5 pt |
| 7 | Nero | Sconfitta | Bianco | Due pass consecutivi | 614 | 57.0–309.5 | -252.5 pt |
| 8 | Bianco | Sconfitta | Nero | Due pass consecutivi | 772 | 306.0–46.5 | -259.5 pt |
| 9 | Nero | Sconfitta | Bianco | Due pass consecutivi | 859 | 57.0–297.5 | -240.5 pt |
| 10 | Bianco | Vittoria | Bianco | Due pass consecutivi | 224 | 178.0–190.5 | +12.5 pt |
| 11 | Nero | Vittoria | Nero | Due pass consecutivi | 272 | 190.0–176.5 | +13.5 pt |
| 12 | Bianco | Vittoria | Bianco | Due pass consecutivi | 236 | 165.0–191.5 | +26.5 pt |
| 13 | Nero | Vittoria | Nero | Due pass consecutivi | 163 | 221.0–147.5 | +73.5 pt |
| 14 | Bianco | Vittoria | Bianco | Due pass consecutivi | 186 | 144.0–223.5 | +79.5 pt |
| 15 | Nero | Vittoria | Nero | Due pass consecutivi | 206 | 212.0–117.5 | +94.5 pt |
| 16 | Bianco | Vittoria | Bianco | Due pass consecutivi | 348 | 178.0–178.5 | +0.5 pt |
| 17 | Nero | Vittoria | Nero | Due pass consecutivi | 431 | 179.0–178.5 | +0.5 pt |
| 18 | Bianco | Vittoria | Bianco | Due pass consecutivi | 232 | 160.0–200.5 | +40.5 pt |
| 19 | Nero | Sconfitta | Bianco | Due pass consecutivi | 606 | 99.0–261.5 | -162.5 pt |
| 20 | Bianco | Vittoria | Bianco | Due pass consecutivi | 238 | 123.0–244.5 | +121.5 pt |
| 21 | Nero | Sconfitta | Bianco | Due pass consecutivi | 458 | 124.0–240.5 | -116.5 pt |
| 22 | Bianco | Vittoria | Bianco | Due pass consecutivi | 476 | 167.0–175.5 | +8.5 pt |
| 23 | Nero | Vittoria | Nero | Due pass consecutivi | 197 | 199.0–167.5 | +31.5 pt |
| 24 | Bianco | Vittoria | Bianco | Due pass consecutivi | 270 | 180.0–180.5 | +0.5 pt |
| 25 | Nero | Sconfitta | Bianco | Due pass consecutivi | 647 | 26.0–331.5 | -305.5 pt |
| 26 | Bianco | Sconfitta | Nero | Due pass consecutivi | 685 | 254.0–114.5 | -139.5 pt |
| 27 | Nero | Vittoria | Nero | Due pass consecutivi | 205 | 213.0–148.5 | +64.5 pt |
| 28 | Bianco | Vittoria | Bianco | Due pass consecutivi | 196 | 173.0–188.5 | +15.5 pt |
| 29 | Nero | Vittoria | Nero | Due pass consecutivi | 145 | 210.0–158.5 | +51.5 pt |
| 30 | Bianco | Vittoria | Bianco | Due pass consecutivi | 250 | 164.0–198.5 | +34.5 pt |
| 31 | Nero | Vittoria | Nero | Due pass consecutivi | 238 | 206.0–160.5 | +45.5 pt |
| 32 | Bianco | Sconfitta | Nero | Due pass consecutivi | 561 | 220.0–135.5 | -84.5 pt |
| 33 | Nero | Vittoria | Nero | Due pass consecutivi | 320 | 185.0–178.5 | +6.5 pt |
| 34 | Bianco | Vittoria | Bianco | Due pass consecutivi | 112 | 143.0–225.5 | +82.5 pt |
| 35 | Nero | Vittoria | Nero | Due pass consecutivi | 269 | 187.0–181.5 | +5.5 pt |
| 36 | Bianco | Vittoria | Bianco | Due pass consecutivi | 128 | 155.0–213.5 | +58.5 pt |
| 37 | Nero | Sconfitta | Bianco | Due pass consecutivi | 715 | 45.0–307.5 | -262.5 pt |
| 38 | Bianco | Vittoria | Bianco | Due pass consecutivi | 180 | 183.0–184.5 | +1.5 pt |
| 39 | Nero | Vittoria | Nero | Due pass consecutivi | 319 | 182.0–181.5 | +0.5 pt |
| 40 | Bianco | Sconfitta | Nero | Due pass consecutivi | 648 | 332.0–31.5 | -300.5 pt |
| 41 | Nero | Sconfitta | Bianco | Due pass consecutivi | 702 | 18.0–343.5 | -325.5 pt |
| 42 | Bianco | Vittoria | Bianco | Due pass consecutivi | 368 | 182.0–182.5 | +0.5 pt |
| 43 | Nero | Sconfitta | Bianco | Due pass consecutivi | 600 | 24.0–339.5 | -315.5 pt |
| 44 | Bianco | Vittoria | Bianco | Due pass consecutivi | 317 | 169.0–194.5 | +25.5 pt |
| 45 | Nero | Vittoria | Nero | Due pass consecutivi | 268 | 183.0–181.5 | +1.5 pt |
| 46 | Bianco | Sconfitta | Nero | Due pass consecutivi | 621 | 261.0–107.5 | -153.5 pt |
| 47 | Nero | Sconfitta | Bianco | Due pass consecutivi | 614 | 90.0–278.5 | -188.5 pt |
| 48 | Bianco | Vittoria | Bianco | Due pass consecutivi | 132 | 178.0–190.5 | +12.5 pt |
| 49 | Nero | Vittoria | Nero | Due pass consecutivi | 383 | 178.0–173.5 | +4.5 pt |
| 50 | Bianco | Sconfitta | Nero | Due pass consecutivi | 717 | 332.0–32.5 | -299.5 pt |
| 51 | Nero | Vittoria | Nero | Due pass consecutivi | 430 | 182.0–178.5 | +3.5 pt |
| 52 | Bianco | Vittoria | Bianco | Due pass consecutivi | 249 | 182.0–182.5 | +0.5 pt |
| 53 | Nero | Sconfitta | Bianco | Due pass consecutivi | 691 | 40.0–313.5 | -273.5 pt |
| 54 | Bianco | Vittoria | Bianco | Due pass consecutivi | 241 | 175.0–193.5 | +18.5 pt |
| 55 | Nero | Sconfitta | Bianco | Due pass consecutivi | 512 | 46.0–322.5 | -276.5 pt |
| 56 | Bianco | Sconfitta | Nero | Due pass consecutivi | 728 | 344.0–18.5 | -325.5 pt |
| 57 | Nero | Sconfitta | Bianco | Due pass consecutivi | 414 | 163.0–200.5 | -37.5 pt |
| 58 | Bianco | Vittoria | Bianco | Due pass consecutivi | 251 | 167.0–195.5 | +28.5 pt |
| 59 | Nero | Vittoria | Nero | Due pass consecutivi | 455 | 179.0–174.5 | +4.5 pt |
| 60 | Bianco | Vittoria | Bianco | Due pass consecutivi | 273 | 171.0–189.5 | +18.5 pt |
| 61 | Nero | Vittoria | Nero | Due pass consecutivi | 444 | 178.0–174.5 | +3.5 pt |
| 62 | Bianco | Vittoria | Bianco | Due pass consecutivi | 221 | 162.0–202.5 | +40.5 pt |
| 63 | Nero | Vittoria | Nero | Due pass consecutivi | 216 | 193.0–175.5 | +17.5 pt |
| 64 | Bianco | Vittoria | Bianco | Due pass consecutivi | 265 | 170.0–195.5 | +25.5 pt |
| 65 | Nero | Sconfitta | Bianco | Due pass consecutivi | 703 | 21.0–339.5 | -318.5 pt |
| 66 | Bianco | Sconfitta | Nero | Due pass consecutivi | 730 | 325.0–32.5 | -292.5 pt |
| 67 | Nero | Sconfitta | Bianco | Due pass consecutivi | 775 | 63.0–284.5 | -221.5 pt |
| 68 | Bianco | Vittoria | Bianco | Due pass consecutivi | 269 | 175.0–188.5 | +13.5 pt |
| 69 | Nero | Vittoria | Nero | Due pass consecutivi | 414 | 178.0–175.5 | +2.5 pt |
| 70 | Bianco | Sconfitta | Nero | Due pass consecutivi | 704 | 310.0–45.5 | -264.5 pt |
| 71 | Nero | Vittoria | Nero | Due pass consecutivi | 213 | 190.0–177.5 | +12.5 pt |
| 72 | Bianco | Vittoria | Bianco | Due pass consecutivi | 385 | 179.0–180.5 | +1.5 pt |
| 73 | Nero | Vittoria | Nero | Due pass consecutivi | 244 | 198.0–169.5 | +28.5 pt |
| 74 | Bianco | Sconfitta | Nero | Due pass consecutivi | 662 | 353.0–12.5 | -340.5 pt |
| 75 | Nero | Sconfitta | Bianco | Due pass consecutivi | 672 | 42.0–317.5 | -275.5 pt |
| 76 | Bianco | Vittoria | Bianco | Due pass consecutivi | 255 | 157.0–203.5 | +46.5 pt |
| 77 | Nero | Vittoria | Nero | Due pass consecutivi | 239 | 194.0–172.5 | +21.5 pt |
| 78 | Bianco | Sconfitta | Nero | Due pass consecutivi | 535 | 302.0–66.5 | -235.5 pt |
| 79 | Nero | Sconfitta | Bianco | Due pass consecutivi | 615 | 19.0–345.5 | -326.5 pt |
| 80 | Bianco | Vittoria | Bianco | Due pass consecutivi | 166 | 116.0–251.5 | +135.5 pt |
| 81 | Nero | Vittoria | Nero | Due pass consecutivi | 290 | 192.0–175.5 | +16.5 pt |
| 82 | Bianco | Vittoria | Bianco | Due pass consecutivi | 268 | 180.0–183.5 | +3.5 pt |
| 83 | Nero | Vittoria | Nero | Due pass consecutivi | 131 | 361.0–7.5 | +353.5 pt |
| 84 | Bianco | Vittoria | Bianco | Due pass consecutivi | 280 | 178.0–181.5 | +3.5 pt |
| 85 | Nero | Sconfitta | Bianco | Due pass consecutivi | 648 | 69.0–299.5 | -230.5 pt |
| 86 | Bianco | Vittoria | Bianco | Due pass consecutivi | 206 | 165.0–199.5 | +34.5 pt |
| 87 | Nero | Sconfitta | Bianco | Due pass consecutivi | 550 | 91.0–277.5 | -186.5 pt |
| 88 | Bianco | Vittoria | Bianco | Due pass consecutivi | 259 | 165.0–186.5 | +21.5 pt |
| 89 | Nero | Sconfitta | Bianco | Due pass consecutivi | 846 | 44.0–312.5 | -268.5 pt |
| 90 | Bianco | Sconfitta | Nero | Due pass consecutivi | 563 | 276.0–87.5 | -188.5 pt |
| 91 | Nero | Vittoria | Nero | Due pass consecutivi | 261 | 182.0–181.5 | +0.5 pt |
| 92 | Bianco | Vittoria | Bianco | Due pass consecutivi | 226 | 147.0–220.5 | +73.5 pt |
| 93 | Nero | Vittoria | Nero | Due pass consecutivi | 248 | 212.0–154.5 | +57.5 pt |
| 94 | Bianco | Vittoria | Bianco | Due pass consecutivi | 234 | 172.0–191.5 | +19.5 pt |
| 95 | Nero | Sconfitta | Bianco | Due pass consecutivi | 693 | 40.0–316.5 | -276.5 pt |
| 96 | Bianco | Vittoria | Bianco | Due pass consecutivi | 387 | 180.0–182.5 | +2.5 pt |
| 97 | Nero | Vittoria | Nero | Due pass consecutivi | 197 | 210.0–158.5 | +51.5 pt |
| 98 | Bianco | Vittoria | Bianco | Due pass consecutivi | 188 | 179.0–189.5 | +10.5 pt |
| 99 | Nero | Vittoria | Nero | Due pass consecutivi | 207 | 191.0–177.5 | +13.5 pt |
| 100 | Bianco | Vittoria | Bianco | Due pass consecutivi | 182 | 106.0–261.5 | +155.5 pt |

## Altri file

- CSV dettagliato: `TRM/model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.csv`
- Full report JSON: `TRM/model_evaluation/results/19x19_go/full_reports/negamax-d2-c3-rank20k-40v-100g.report.json`

