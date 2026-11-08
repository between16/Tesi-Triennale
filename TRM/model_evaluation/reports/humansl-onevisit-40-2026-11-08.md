# Valutazione HumanSL a una visita: 40 partite per profilo

## Protocollo

- Checkpoint TRM: i checkpoint migliori del progetto per 9×9 e 19×19.
- Avversario: KataGo v1.18.2, rete principale appropriata alla dimensione
  del goban e rete ufficiale HumanSL; un profilo per serie.
- Profili: `rank_1k`, `rank_5k`, `rank_10k`, `rank_15k`, `rank_20k`.
- Limite di ricerca: `maxVisits=1`.
- Campionamento HumanSL: impostazioni di campionamento ampio della
  configurazione ufficiale `gtp_human5k_example.cfg` di KataGo v1.18.2:
  temperatura 0,85 in apertura, 0,70 nelle fasi successive, decadimento
  dimezzato ogni 80 mosse e temperatura applicata solo alle mosse con
  probabilità inferiore all'1%.
- Colori: 20 partite con il TRM Nero e 20 con il TRM Bianco per ogni profilo
  e formato. Il colore del TRM alterna a ogni partita.
- Regole: area scoring, komi 7,5, positional superko, suicidio vietato,
  nessun button/tax e partita conclusa dopo due pass consecutivi.
- Correzione PASS: configurazione predefinita TRM, che sottrae 3,5 al logit
  PASS prima che siano occupate 16 intersezioni.
- Tutte le 400 partite sono state completate; non ci sono stati pareggi.

In questa modalità HumanSL, `humanSLChosenMoveProp=1.0` e un valore molto alto
di `humanSLChosenMovePiklLambda` fanno sì che la policy HumanSL domini la
selezione delle mosse. La documentazione KataGo precisa che, in tale
configurazione, aumentare le visite incide poco sulla mossa scelta, salvo
potenzialmente sul giudizio di pass/resign. Qui la scelta dei pass è delegata
a KataGo (`humanSLChosenMoveIgnorePass=true`) e la resa è disabilitata.
Pertanto una visita è il limite di ricerca richiesto, ma non costituisce da
solo una misura generale della forza di KataGo.

## Risultati

Vittorie e sconfitte sono viste dal punto di vista del TRM. Bianco e Nero
indicano il colore assegnato al TRM; per ciascun profilo ci sono 20 partite
per colore.

| Goban | Profilo HumanSL | Vittorie TRM Bianco | Vittorie TRM Nero | Sconfitte TRM Bianco | Sconfitte TRM Nero | Totale |
|---|---|---:|---:|---:|---:|---:|
| 9×9 | rank_1k | 5 | 0 | 15 | 20 | 5/40 (12,5%) |
| 9×9 | rank_5k | 5 | 3 | 15 | 17 | 8/40 (20,0%) |
| 9×9 | rank_10k | 7 | 1 | 13 | 19 | 8/40 (20,0%) |
| 9×9 | rank_15k | 13 | 2 | 7 | 18 | 15/40 (37,5%) |
| 9×9 | rank_20k | 15 | 5 | 5 | 15 | 20/40 (50,0%) |
| 19×19 | rank_1k | 0 | 0 | 20 | 20 | 0/40 (0,0%) |
| 19×19 | rank_5k | 0 | 1 | 20 | 19 | 1/40 (2,5%) |
| 19×19 | rank_10k | 3 | 2 | 17 | 18 | 5/40 (12,5%) |
| 19×19 | rank_15k | 1 | 1 | 19 | 19 | 2/40 (5,0%) |
| 19×19 | rank_20k | 8 | 4 | 12 | 16 | 12/40 (30,0%) |

Intervalli di confidenza Wilson al 95% per il tasso totale di vittoria:

| Goban | Profilo HumanSL | Tasso vittorie | Intervallo Wilson al 95% |
|---|---|---:|---:|
| 9×9 | rank_1k | 12,5% | 5,5%–26,1% |
| 9×9 | rank_5k | 20,0% | 10,5%–34,8% |
| 9×9 | rank_10k | 20,0% | 10,5%–34,8% |
| 9×9 | rank_15k | 37,5% | 24,2%–53,0% |
| 9×9 | rank_20k | 50,0% | 35,2%–64,8% |
| 19×19 | rank_1k | 0,0% | 0,0%–8,8% |
| 19×19 | rank_5k | 2,5% | 0,4%–12,9% |
| 19×19 | rank_10k | 12,5% | 5,5%–26,1% |
| 19×19 | rank_15k | 5,0% | 1,4%–16,5% |
| 19×19 | rank_20k | 30,0% | 18,1%–45,4% |

## Interpretazione e limiti

- Sul 9×9 il TRM ha diviso le partite 20–20 contro `rank_20k`; tuttavia, le
  vittorie sono state 15/20 da Bianco e 5/20 da Nero. Servono più partite per
  capire se questa differenza per colore sia stabile.
- Sul 19×19 il TRM non ha raggiunto un risultato pari con nessuno dei profili
  testati. Il risultato migliore è stato 12/40 contro `rank_20k`, con 8
  vittorie da Bianco e 4 da Nero.
- I risultati non crescono in modo monotono a ogni profilo: sul 19×19
  `rank_10k` ha ottenuto un risultato migliore di `rank_15k`. Con 40 partite
  per condizione la variabilità campionaria è rilevante.
- Le etichette HumanSL descrivono profili di imitazione delle mosse; non sono
  rank KGS/AGA calibrati né rating Elo misurati. I risultati valgono solo
  contro le specifiche configurazioni locali qui utilizzate.
- Per i due formati sono state utilizzate reti principali KataGo diverse.
  Pertanto il confronto fra 9×9 e 19×19 non isola l'effetto della sola
  dimensione del goban.
- Questi dati sono evidenza esplorativa sui match, non dimostrano un rank
  assoluto né una stima Elo precisa.

## File dei risultati

I dieci CSV con le singole partite e il riepilogo leggibile dalle macchine
sono in `../results/human-profile-40-2026-11-08/`. Ogni CSV registra profilo,
visite, modalità di temperatura, regole, colore del TRM, risultato,
punteggio e statistiche PASS.
