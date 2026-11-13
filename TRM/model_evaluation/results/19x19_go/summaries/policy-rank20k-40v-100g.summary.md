# Riepilogo test TRM vs KataGo — 19x19

- **Stato:** COMPLETATO
- **Inizio (UTC):** 2026-11-13 20:14:12
- **Fine (UTC):** 2026-11-13 20:31:46
- **Durata:** 17.6 minuti (1054.7 secondi)
- **Avversario:** KataGo 1.18.2+tf3-b11c768-s12382M-d6533M+b18c384nbt-humanv0
- **Modalità TRM:** policy

## Parametri del test

Comando eseguito:
```bash
python3 model_evaluation/go_match.py --board-size 19 --katago model_evaluation/katago-local/engine/squashfs-root/usr/bin/katago --katago-config model_evaluation/katago-local/engine/default_gtp.cfg --katago-model model_evaluation/katago-local/downloads/kata1-tf3-b11c768-s12382M-d6533M.bin.gz --katago-human-model model_evaluation/katago-local/downloads/b18c384nbt-humanv0.bin.gz --human-profile rank_20k --games 100 --max-visits 40 --seed 42 --pass-logit-penalty 3.5 --pass-penalty-before-stones 16 --search policy --output model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.csv
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
| --search | policy |
| --search-depth | 2 |
| --search-candidates | — |
| --max-visits | 40 |
| --gtp-timeout | 600.0 |
| --device | auto |
| --pass-logit-penalty | 3.5 |
| --pass-penalty-before-stones | 16 |
| --seed | 42 |
| --output | TRM/model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.csv |
| --report-output | TRM/model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.report.json |
| --summary-output | TRM/model_evaluation/results/19x19_go/summaries/policy-rank20k-40v-100g.summary.md |

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
| Profondità Negamax effettiva | 0 |
| CSV partite | TRM/model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.csv |
| Full report JSON | TRM/model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.report.json |
| Riepilogo Markdown | TRM/model_evaluation/results/19x19_go/summaries/policy-rank20k-40v-100g.summary.md |

## Risultati complessivi

| Indicatore | Risultato |
| --- | --- |
| Partite richieste | 100 |
| Partite registrate | 100 |
| Partite completate | 100 |
| Partite incomplete | 0 |
| Partite non iniziate | 0 |
| Vittorie TRM | 48 |
| Pareggi | 0 |
| Sconfitte TRM (vittorie KataGo) | 52 |
| Win rate TRM | 48.0% |
| Score rate TRM (vittoria=1, pareggio=0.5) | 48.0% |
| Intervallo 95% win rate (Wilson) | 38.5%–57.7% |
| Gap medio TRM | -1.3 pt |
| Gap mediano TRM | -0.5 pt |
| Gap medio assoluto | 39.6 pt |
| Partite con punteggio numerico | 100 |
| Gap a favore TRM / a favore KataGo / pari | 48 / 52 / 0 |
| Mosse medie per partita registrata | 216.5 |
| PASS giocati dal TRM | 115 |
| PASS prima di 16 pietre | 0 |
| Decisioni con penalità PASS attiva | 800 |
| Terminazioni | two_passes: 100 |

*Il gap è calcolato dal punto di vista del TRM: un valore positivo indica un vantaggio del TRM; un valore negativo indica un vantaggio di KataGo. Il gap è disponibile solo per le partite terminate con un punteggio numerico.*

## Risultati per colore del TRM

| Colore TRM | Giocate | Completate | V-P-S | Win rate | Gap medio |
| --- | --- | --- | --- | --- | --- |
| Nero | 50 | 50 | 22-0-28 | 44.0% | -1.6 pt |
| Bianco | 50 | 50 | 26-0-24 | 52.0% | -1.0 pt |

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
| 1 | Nero | Sconfitta | Bianco | Due pass consecutivi | 217 | 173.0–195.5 | -22.5 pt |
| 2 | Bianco | Sconfitta | Nero | Due pass consecutivi | 241 | 203.0–158.5 | -44.5 pt |
| 3 | Nero | Sconfitta | Bianco | Due pass consecutivi | 249 | 141.0–227.5 | -86.5 pt |
| 4 | Bianco | Vittoria | Bianco | Due pass consecutivi | 138 | 174.0–194.5 | +20.5 pt |
| 5 | Nero | Vittoria | Nero | Due pass consecutivi | 181 | 192.0–176.5 | +15.5 pt |
| 6 | Bianco | Sconfitta | Nero | Due pass consecutivi | 202 | 132.0–131.5 | -0.5 pt |
| 7 | Nero | Vittoria | Nero | Due pass consecutivi | 234 | 187.0–176.5 | +10.5 pt |
| 8 | Bianco | Vittoria | Bianco | Due pass consecutivi | 266 | 163.0–190.5 | +27.5 pt |
| 9 | Nero | Vittoria | Nero | Due pass consecutivi | 129 | 238.0–130.5 | +107.5 pt |
| 10 | Bianco | Vittoria | Bianco | Due pass consecutivi | 148 | 125.0–241.5 | +116.5 pt |
| 11 | Nero | Vittoria | Nero | Due pass consecutivi | 204 | 249.0–119.5 | +129.5 pt |
| 12 | Bianco | Vittoria | Bianco | Due pass consecutivi | 212 | 150.0–218.5 | +68.5 pt |
| 13 | Nero | Sconfitta | Bianco | Due pass consecutivi | 265 | 163.0–205.5 | -42.5 pt |
| 14 | Bianco | Sconfitta | Nero | Due pass consecutivi | 269 | 195.0–169.5 | -25.5 pt |
| 15 | Nero | Vittoria | Nero | Due pass consecutivi | 203 | 200.0–167.5 | +32.5 pt |
| 16 | Bianco | Sconfitta | Nero | Due pass consecutivi | 312 | 185.0–178.5 | -6.5 pt |
| 17 | Nero | Vittoria | Nero | Due pass consecutivi | 165 | 187.0–181.5 | +5.5 pt |
| 18 | Bianco | Sconfitta | Nero | Due pass consecutivi | 226 | 206.0–162.5 | -43.5 pt |
| 19 | Nero | Sconfitta | Bianco | Due pass consecutivi | 299 | 164.0–203.5 | -39.5 pt |
| 20 | Bianco | Sconfitta | Nero | Due pass consecutivi | 311 | 258.0–110.5 | -147.5 pt |
| 21 | Nero | Sconfitta | Bianco | Due pass consecutivi | 233 | 183.0–183.5 | -0.5 pt |
| 22 | Bianco | Vittoria | Bianco | Due pass consecutivi | 194 | 179.0–189.5 | +10.5 pt |
| 23 | Nero | Sconfitta | Bianco | Due pass consecutivi | 203 | 182.0–186.5 | -4.5 pt |
| 24 | Bianco | Sconfitta | Nero | Due pass consecutivi | 247 | 192.0–175.5 | -16.5 pt |
| 25 | Nero | Vittoria | Nero | Due pass consecutivi | 165 | 266.0–102.5 | +163.5 pt |
| 26 | Bianco | Vittoria | Bianco | Due pass consecutivi | 164 | 173.0–195.5 | +22.5 pt |
| 27 | Nero | Sconfitta | Bianco | Due pass consecutivi | 247 | 170.0–185.5 | -15.5 pt |
| 28 | Bianco | Vittoria | Bianco | Due pass consecutivi | 152 | 170.0–198.5 | +28.5 pt |
| 29 | Nero | Sconfitta | Bianco | Due pass consecutivi | 191 | 177.0–191.5 | -14.5 pt |
| 30 | Bianco | Vittoria | Bianco | Due pass consecutivi | 150 | 175.0–193.5 | +18.5 pt |
| 31 | Nero | Vittoria | Nero | Due pass consecutivi | 165 | 213.0–155.5 | +57.5 pt |
| 32 | Bianco | Vittoria | Bianco | Due pass consecutivi | 172 | 161.0–207.5 | +46.5 pt |
| 33 | Nero | Vittoria | Nero | Due pass consecutivi | 249 | 182.0–177.5 | +4.5 pt |
| 34 | Bianco | Vittoria | Bianco | Due pass consecutivi | 152 | 170.0–198.5 | +28.5 pt |
| 35 | Nero | Vittoria | Nero | Due pass consecutivi | 195 | 190.0–167.5 | +22.5 pt |
| 36 | Bianco | Sconfitta | Nero | Due pass consecutivi | 252 | 206.0–162.5 | -43.5 pt |
| 37 | Nero | Sconfitta | Bianco | Due pass consecutivi | 239 | 176.0–191.5 | -15.5 pt |
| 38 | Bianco | Sconfitta | Nero | Due pass consecutivi | 267 | 199.0–166.5 | -32.5 pt |
| 39 | Nero | Sconfitta | Bianco | Due pass consecutivi | 213 | 156.0–187.5 | -31.5 pt |
| 40 | Bianco | Sconfitta | Nero | Due pass consecutivi | 279 | 235.0–125.5 | -109.5 pt |
| 41 | Nero | Sconfitta | Bianco | Due pass consecutivi | 264 | 148.0–220.5 | -72.5 pt |
| 42 | Bianco | Sconfitta | Nero | Due pass consecutivi | 228 | 198.0–170.5 | -27.5 pt |
| 43 | Nero | Sconfitta | Bianco | Due pass consecutivi | 181 | 180.0–188.5 | -8.5 pt |
| 44 | Bianco | Sconfitta | Nero | Due pass consecutivi | 188 | 202.0–166.5 | -35.5 pt |
| 45 | Nero | Vittoria | Nero | Due pass consecutivi | 169 | 205.0–163.5 | +41.5 pt |
| 46 | Bianco | Vittoria | Bianco | Due pass consecutivi | 180 | 173.0–195.5 | +22.5 pt |
| 47 | Nero | Sconfitta | Bianco | Due pass consecutivi | 281 | 166.0–202.5 | -36.5 pt |
| 48 | Bianco | Vittoria | Bianco | Due pass consecutivi | 214 | 150.0–217.5 | +67.5 pt |
| 49 | Nero | Sconfitta | Bianco | Due pass consecutivi | 262 | 175.0–191.5 | -16.5 pt |
| 50 | Bianco | Sconfitta | Nero | Due pass consecutivi | 238 | 180.0–175.5 | -4.5 pt |
| 51 | Nero | Sconfitta | Bianco | Due pass consecutivi | 205 | 176.0–178.5 | -2.5 pt |
| 52 | Bianco | Vittoria | Bianco | Due pass consecutivi | 225 | 163.0–205.5 | +42.5 pt |
| 53 | Nero | Vittoria | Nero | Due pass consecutivi | 187 | 184.0–182.5 | +1.5 pt |
| 54 | Bianco | Sconfitta | Nero | Due pass consecutivi | 194 | 167.0–115.5 | -51.5 pt |
| 55 | Nero | Sconfitta | Bianco | Due pass consecutivi | 271 | 152.0–215.5 | -63.5 pt |
| 56 | Bianco | Vittoria | Bianco | Due pass consecutivi | 180 | 179.0–189.5 | +10.5 pt |
| 57 | Nero | Sconfitta | Bianco | Due pass consecutivi | 235 | 155.0–213.5 | -58.5 pt |
| 58 | Bianco | Sconfitta | Nero | Due pass consecutivi | 220 | 191.0–177.5 | -13.5 pt |
| 59 | Nero | Sconfitta | Bianco | Due pass consecutivi | 260 | 166.0–199.5 | -33.5 pt |
| 60 | Bianco | Vittoria | Bianco | Due pass consecutivi | 196 | 163.0–202.5 | +39.5 pt |
| 61 | Nero | Sconfitta | Bianco | Due pass consecutivi | 227 | 173.0–194.5 | -21.5 pt |
| 62 | Bianco | Vittoria | Bianco | Due pass consecutivi | 232 | 170.0–198.5 | +28.5 pt |
| 63 | Nero | Sconfitta | Bianco | Due pass consecutivi | 247 | 163.0–201.5 | -38.5 pt |
| 64 | Bianco | Sconfitta | Nero | Due pass consecutivi | 230 | 223.0–143.5 | -79.5 pt |
| 65 | Nero | Vittoria | Nero | Due pass consecutivi | 161 | 193.0–175.5 | +17.5 pt |
| 66 | Bianco | Sconfitta | Nero | Due pass consecutivi | 231 | 190.0–177.5 | -12.5 pt |
| 67 | Nero | Vittoria | Nero | Due pass consecutivi | 153 | 211.0–157.5 | +53.5 pt |
| 68 | Bianco | Sconfitta | Nero | Due pass consecutivi | 242 | 211.0–152.5 | -58.5 pt |
| 69 | Nero | Vittoria | Nero | Due pass consecutivi | 227 | 188.0–180.5 | +7.5 pt |
| 70 | Bianco | Sconfitta | Nero | Due pass consecutivi | 281 | 198.0–152.5 | -45.5 pt |
| 71 | Nero | Sconfitta | Bianco | Due pass consecutivi | 254 | 120.0–234.5 | -114.5 pt |
| 72 | Bianco | Sconfitta | Nero | Due pass consecutivi | 228 | 201.0–167.5 | -33.5 pt |
| 73 | Nero | Vittoria | Nero | Due pass consecutivi | 215 | 218.0–150.5 | +67.5 pt |
| 74 | Bianco | Vittoria | Bianco | Due pass consecutivi | 152 | 153.0–215.5 | +62.5 pt |
| 75 | Nero | Sconfitta | Bianco | Due pass consecutivi | 258 | 176.0–191.5 | -15.5 pt |
| 76 | Bianco | Vittoria | Bianco | Due pass consecutivi | 226 | 148.0–165.5 | +17.5 pt |
| 77 | Nero | Vittoria | Nero | Due pass consecutivi | 167 | 206.0–159.5 | +46.5 pt |
| 78 | Bianco | Vittoria | Bianco | Due pass consecutivi | 240 | 169.0–197.5 | +28.5 pt |
| 79 | Nero | Sconfitta | Bianco | Due pass consecutivi | 213 | 177.0–191.5 | -14.5 pt |
| 80 | Bianco | Vittoria | Bianco | Due pass consecutivi | 202 | 165.0–201.5 | +36.5 pt |
| 81 | Nero | Sconfitta | Bianco | Due pass consecutivi | 167 | 184.0–184.5 | -0.5 pt |
| 82 | Bianco | Sconfitta | Nero | Due pass consecutivi | 229 | 202.0–165.5 | -36.5 pt |
| 83 | Nero | Sconfitta | Bianco | Due pass consecutivi | 247 | 181.0–187.5 | -6.5 pt |
| 84 | Bianco | Sconfitta | Nero | Due pass consecutivi | 245 | 194.0–174.5 | -19.5 pt |
| 85 | Nero | Vittoria | Nero | Due pass consecutivi | 199 | 191.0–177.5 | +13.5 pt |
| 86 | Bianco | Vittoria | Bianco | Due pass consecutivi | 228 | 157.0–211.5 | +54.5 pt |
| 87 | Nero | Vittoria | Nero | Due pass consecutivi | 219 | 189.0–179.5 | +9.5 pt |
| 88 | Bianco | Vittoria | Bianco | Due pass consecutivi | 138 | 170.0–198.5 | +28.5 pt |
| 89 | Nero | Vittoria | Nero | Due pass consecutivi | 173 | 204.0–164.5 | +39.5 pt |
| 90 | Bianco | Vittoria | Bianco | Due pass consecutivi | 282 | 181.0–184.5 | +3.5 pt |
| 91 | Nero | Vittoria | Nero | Due pass consecutivi | 187 | 210.0–158.5 | +51.5 pt |
| 92 | Bianco | Vittoria | Bianco | Due pass consecutivi | 197 | 147.0–169.5 | +22.5 pt |
| 93 | Nero | Sconfitta | Bianco | Due pass consecutivi | 259 | 159.0–209.5 | -50.5 pt |
| 94 | Bianco | Sconfitta | Nero | Due pass consecutivi | 269 | 197.0–140.5 | -56.5 pt |
| 95 | Nero | Vittoria | Nero | Due pass consecutivi | 185 | 193.0–168.5 | +24.5 pt |
| 96 | Bianco | Vittoria | Bianco | Due pass consecutivi | 170 | 142.0–226.5 | +84.5 pt |
| 97 | Nero | Sconfitta | Bianco | Due pass consecutivi | 201 | 176.0–192.5 | -16.5 pt |
| 98 | Bianco | Vittoria | Bianco | Due pass consecutivi | 203 | 157.0–209.5 | +52.5 pt |
| 99 | Nero | Sconfitta | Bianco | Due pass consecutivi | 250 | 104.0–263.5 | -159.5 pt |
| 100 | Bianco | Sconfitta | Nero | Due pass consecutivi | 242 | 232.0–135.5 | -96.5 pt |

## Altri file

- CSV dettagliato: `TRM/model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.csv`
- Full report JSON: `TRM/model_evaluation/results/19x19_go/full_reports/policy-rank20k-40v-100g.report.json`

