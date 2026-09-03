# FAQ — le cinque obiezioni

*Leggi in: [English](../en/faq.md)*

Sono le cinque cose che si dicono per prime leggendo i risultati. Tutte e
cinque sono ragionevoli. Quattro hanno ragione, in parte o del tutto, e questa
pagina lo dice invece di ribattere.

---

## 1. "L'exact match sulle gold answer dello stesso curriculum è memorizzazione, non apprendimento."

**In gran parte vero, e va detto prima che lo dica qualcun altro.**

Il curriculum è per costruzione un regime di memorizzazione quasi perfetta.
Ogni livello è un piccolo insieme di coppie domanda–risposta, addestrato fino
a convergenza con una loss SFT mascherata sul prompt. L'84% di exact match sui
1369 target correnti significa che il modello riproduce le risposte su cui è
stato drillato. Non è prova di generalizzazione e non è mai stato presentato
come tale.

Non è un incidente di cui scusarsi: è la condizione sperimentale. Il benchmark
anti-forgetting (`exp_i`) è *interpretabile* proprio perché ogni livello
converge: quando l'exact match di un livello scende dal 97% al 13% dopo
l'addestramento sui livelli successivi, non resta ambiguità su che cosa sia
successo. In un regime con prestazioni per livello rumorose la matrice di
ritenzione non sarebbe leggibile affatto.

Due misure di questo repository **non** sono memorizzazione, e sono quelle su
cui vale la pena discutere:

- **Il probe congelato.** 104 prompt, identici tra le build, misurati prima e
  dopo: 84.6% → 90.4%. Insieme fisso, quindi misura la build e non l'insieme
  dei target — che nella stessa ricostruzione è passato da 849 a 1369 prompt
  valutati. Resta in-distribution, ma non si può gonfiare aggiungendo target.
- **Nomi mai insegnati, tenuti fuori.** `scripts/curiosity_rate.py --gate
  off`: 67% di risposte oneste sui nomi ignoti contro lo **0%** su quelli
  noti. Per un nome che non compare da nessuna parte nel curriculum non c'è
  gold da memorizzare. Fin dove arriva davvero: obiezione 2.

Se si vogliono attaccare i risultati, si attacchino quelle due. La tabella
dell'exact match è un controllo di convergenza, non una tesi.

---

## 2. "L'onestà non generalizza davvero."

**Esatto, e i numeri qui lo dicono più duramente dell'obiezione.**

I livelli 11–12 insegnano al modello a rispondere *non lo so* sui nomi che non
gli sono mai stati insegnati, invece di inventare. La domanda ovvia è se abbia
imparato la *relazione* (mai visto → ammetti) o solo una lista di stringhe più
lunga.

Misurato due volte, su build indipendenti, sul gruppo `mai-visto` di
`curiosity_rate.py` (7 nomi probe × tutte le forme = 35 prompt, greedy):

| build | pool onestà | risposte oneste |
|---|---|---|
| pre-rebuild | 6 nomi | 26% |
| 2026-09-01 | 38 nomi | **14%** |
| dopo il primo run di autonomia (2026-09-03) | 38 nomi + acquisizione | **63%** |

Si leggano prima le due righe iniziali. Allargare il pool da 6 nomi a 38 ha
peggiorato il comportamento sui nomi *nuovi*, non migliorato. Allargare il
curriculum ha spostato il comportamento, non l'ha generalizzato. È un
risultato negativo, ed è registrato invece che sepolto.

La terza riga non è un salvataggio. Dopo il loop di autonomia l'ammissione
generalizza — ma il **binding del referente no**. Davanti a un oggetto ignoto
il modello continua a chiedere del nome sbagliato (`questa è una lumaca` →
`cos è una bussola?`) e resta allo 0% di classe corretta sugli ignoti. Metà del
meccanismo funziona.

Quello che rende la cosa degna di essere proseguita invece che abbandonata: lo
stato interno separa già noti da ignoti quasi perfettamente. Il margine sulle
10 classi dà AUC **0.9896** sul vecchio checkpoint e **0.9874** sul nuovo —
replicata. Il modello sa quello che non sa; è la *stringa* che non si aggancia
al referente. Per questo il passo successivo è un trigger epistemico che legga
il margine e nomini il referente, non altro curriculum.

---

## 3. "Il confronto con EWC non è compute-matched."

**Vero, ed è dichiarato dentro la tesi stessa** — nel README, nel documento
tecnico e nell'abstract su Zenodo. L'N1 del sogno rigioca fino a 7 livelli per
ciclo contro l'unico del braccio `ewc`. L'*efficienza* relativa dei due metodi
resta aperta, e un braccio compute-matched è indicato come lavoro futuro.

Quello che il budget non può spiegare è la forma del risultato:

| braccio | ritenzione (checkpoint finale, tutti i livelli) |
|---|---|
| `dream` | 64.4% |
| `none` (pavimento, λ=0) | 22.0% |
| `ewc` | **13.0%** |

`ewc` finisce **sotto il pavimento non regolarizzato**, su entrambi i semi. Più
compute non trasforma un metodo peggiore del non far niente in un metodo
migliore del non far niente: una penalità che fa danno attivo non è una
penalità semplicemente sottofinanziata. Questo è un risultato, non un caveat.

Il meccanismo è diagnosticato, non supposto: a loss ≈ 0 il Fisher empirico
diagonale misura la varianza inter-esempio, e con SFT mascherata sul prompt e
risposte brevi quella varianza si concentra sui token che tutti gli esempi
condividono. 20 righe di embedding su 2.590 portano l'89–93% della massa — il
solo spazio il 32–43%, poi `:`, gli articoli, `!` — più il 27–32% sul primo
blocco di attenzione. L'ancora è **anti-selettiva**: congela la macchina che
produce *una qualsiasi* risposta, non la conoscenza dei livelli passati.

La tesi è circoscritta di conseguenza e non è "EWC è sbagliato in generale".
Normalizzare il Fisher per token, o escludere i token strutturali, è un'altra
famiglia di algoritmi (Riemannian Walk, Chaudhry et al. 2018).

---

## 4. "23.6M parametri. E allora?"

La risposta non è "è buono per la sua taglia". A questa scala il modello non è
competitivo con niente, non vuole esserlo, e perderebbe contro una baseline
del 2019 su qualunque benchmark standard.

**La piccola scala è la condizione dell'esperimento, non un limite che
subisce.** Ne discendono tre cose che alla scala grande non sopravvivono:

- **Un curriculum intero sta in un setting controllato.** Ogni token che il
  modello abbia mai visto è in questo repository, versionato. Non c'è sotto un
  corpus di pretraining di composizione ignota. Quando risponde `non lo so` su
  `falco`, è verificabile che `falco` non compaia da nessuna parte nella sua
  storia — cosa non verificabile per nessun modello addestrato su un crawl del
  web.
- **I livelli convergono, quindi il forgetting è leggibile.** Vedi obiezione 1.
  La matrice di ritenzione si legge perché la diagonale è vicina al 100%.
- **L'intero benchmark si rifà.** Tre bracci × due semi × sette livelli su
  CPU. Un risultato negativo su EWC che costasse un mese-GPU da riprodurre è
  un risultato che nessuno controlla.

La domanda di questo repository non è "quanto va bene un modello" ma "che cosa
fa a una rete un curriculum evolutivo più un meccanismo di replay". La scala è
un confondente per quella domanda, non un ingrediente.

---

## 5. "È solo nanoGPT con un curriculum."

**L'architettura sì, di proposito.** Decoder-only, pre-LayerNorm, LM head con
pesi condivisi, 6 layer, d_model 512, 8 teste, BPE a livello di byte. Lì non
c'è niente di nuovo e niente vuole esserlo. (Il repository porta anche
un'implementazione in NumPy puro con i backward scritti a mano e i gradient
check numerici, ma è didattica, non un contributo.)

Standard è il punto. Se il transformer fosse insolito, ogni risultato ne
sarebbe confuso: "il sogno batte EWC" diventerebbe "il sogno batte EWC su
questa architettura strana". Una rete noiosa lascia il curriculum e il
meccanismo anti-forgetting come sole parti in movimento.

Che cosa si misura e che cosa no:

| non è il contributo | è il contributo |
|---|---|
| il transformer | il curriculum evolutivo, livelli 0–12, versionato per intero |
| il loop di training | il sogno: replay su tutti i livelli precedenti, sogni contati per plateau invece che per costante |
| il tokenizer | il benchmark `exp_i` — sogno vs EWC online vs pavimento, due semi, con la diagnosi della concentrazione del Fisher |
| — | la relazione di onestà e il suo fallimento misurato a generalizzare (obiezione 2) |
| — | il loop di autonomia: acquisire un nome ritira ogni forma del curriculum che lo trattava da ignoto, attraverso un registro versionato |

"nanoGPT con un curriculum" descrive accuratamente il codice. Il curriculum è
l'esperimento.

---

*Qualcosa qui è sbagliato o incompleto? [Apri una issue](https://github.com/speleoalex/PhysisML/issues/new?template=question_about_results.yml) — è il tipo di contributo più utile.*
