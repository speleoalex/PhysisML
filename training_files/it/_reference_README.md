# `_reference/` — testi presenti ma non usati per l'addestramento

Ogni livello può avere una sottocartella `_reference/`. I testi che stanno lì
sono conservati per riproducibilità e per la nota di licenza del README, ma
**non entrano in nessuna fase**: né nel training testuale (phase 0), né
nell'insegnamento (phase 1), né nel replay del sogno (N1), né nella
costruzione del tokenizer.

Il meccanismo è la posizione, non un elenco: tutti i caricatori usano un glob
`*.txt` non ricorsivo sulla cartella del livello, quindi una sottocartella è
invisibile per costruzione.

## Perché

Dal livello 3 in su esisteva già un filtro che salta i file oltre 100KB, ma
era una protezione **incidentale**: dipendeva dalla dimensione del file. Un
estratto ridotto sotto soglia sarebbe rientrato nel training senza che nulla
lo segnalasse, e il tokenizer li leggeva comunque tutti, soglia o no.

Due ragioni per tenerli fuori:

- **La prosa per adulti cancella le associazioni prompt→risposta.** Misurato:
  reintrodurre corpora narrativi non filtrati come ultimo atto di training
  distruggeva il 44% di quanto la fase supervisionata aveva appena costruito.
- **La lingua non è quella del curriculum.** La *Divina Commedia* usa italiano
  arcaico; i sottotitoli contengono registri e argomenti da adulti.

## Cosa contengono

| Livello | File | Note |
|---------|------|------|
| 3 | `opensubtitles_L3.txt`, `pinocchio.txt` | 21MB di sottotitoli |
| 4 | `opensubtitles_L4.txt` | 26MB |
| 5 | `opensubtitles_L5.txt`, `de_amicis_ricordi.txt`, `canzoni_moderne.txt` | 83MB |
| 6 | `neera_indomani.txt`, `serao_infedele.txt` | narrativa adulta |
| 8 | `promessi_sposi_estratto.txt` | |
| 9 | `promessi_sposi.txt` | 1.4MB |
| 10 | `divina_commedia.txt` | italiano arcaico |

Su cosa il curriculum si addestra davvero, per ogni livello: `qa_corpus.txt`
(coppie prompt→risposta dalle sessioni) più un testo curato e breve
(`frasi_*.txt`, `favole_esopo.txt`, …).

## Se un testo serve

Non spostarlo indietro: aggiungi al livello un estratto curato e coerente col
livello, sotto i 100KB. Un `_reference/` che si svuota da solo è il segnale
che il curriculum ha smesso di dipendere da materiale non adatto.

I termini di licenza di questi testi restano quelli delle rispettive fonti —
vedi la sezione Licenza del README.
