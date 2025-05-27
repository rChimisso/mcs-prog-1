# Relazione

##### *2025, Metodi del Calcolo Scientifico, Riccardo Chimisso 866009 & Mauro Zorzin 866001*

## Struttura

*La struttura di questa libreria è molto semplice e descritta di seguito.*  
*Una descrizione più dettagliata di ciascun modulo, classe e metodo è sempre disponibile per l'ultima release all'indirizzo https://rchimisso.github.io/mcs-prog-1/.*

### Codice

Il codice sorgente di questa libreria è diviso in due moduli principali: `solver.py` e `main.py`.

#### `solvers.py`

Questo è il modulo che contiene il codice dei diversi risolutori.

La classe astratta `IterativeSolver` fornisce un'interfaccia comune.  
Questo permette di riutilizzare codice, sia internamente, come per il calcolo dell'errore residuo, sia esternamente, nel caso si voglia astrarre la logica applicativa dall'implementazione dello specifico risolutore.

Grazie alla classe astratta, ciascun risolutore può essere inizializzato con valori per tolleranza (`tol`) e numero massimo di iterazioni (`max_iter`) di default. I valori passati durante l'istanziazione del risolutore verranno usati qualora non vengano forniti nella chiamata al metodo `solve`.

Il metodo `solve` richiede di passare la matrice $A$ e il vettore $b$.  
Opzionalmente è possibile specificare una soluzione di partenza $x_0$ diversa dal vettore con tutti $0$, così come una tolleranza o un numero massimo di iterazioni diversi da quelli di base del risolutore.  
Il risultato restituito è una tupla con la soluzione approssimata trovata e i dati raccolti sull'esecuzione.

I 4 risolutori disponibili sono classi concrete che ereditano da `IterativeSolver`: `JacobiSolver`, `GaussSeidelSolver`, `GradientDescentSolver` e `ConjugateGradientSolver`.  
Ciascun risolutore ha nel nome l'algoritmo risolutivo che implementa.

È inoltre presente la *dataclass* `Info` che racchiude i dati per un'esecuzione di un risolutore, permettendo di disaccoppiare i risultati degli algoritmi da quelli di stampa.

È importante sottolineare che tutta la matematica gira in `float64` e i metodi presuppongono matrici SPD (simmetriche e definite positive).

#### `main.py`

Questo è il modulo principale per l'utilizzo della libreria a linea di comando.

Contiene la logica per il parsing degli argomenti a linea di comando, la lettura di file `.mtx`, l'esecuzione dei diversi risolutori coi parametri specificati e la stampa dei relativi risultati.

### Dati

Sono presenti alcuni file `.mtx` di base come esempio da poter usare per confrontare i diversi risolutori.

I file il cui nome termina in `x`, escludendo l'estensione, sono file che rappresentano un vettore con ogni componente a $1$ per il relativo file che rappresenta invece una matrice.  
Ad esempio, il file `spa1.mtx` è una matrice, mentre il file `spa1x.mtx` è il vettore con tutti $1$ associato.  
Questi file aggiuntivi sono forniti come esempio per il parametro opzionale `x` del programma, il quale permette di specificare un vettore soluzione esatta per il sistema.

### Test

La libreria include un semplice file di test che verifica, per un'esecuzione su ogni matrice d'esempio, che tutti i risolutori rispettino le indicazioni fornite tramite parametri e che convergano.

### Documentazione

La libreria fornisce anche della documentazione dettagliata, oltre a questa relazione.

La documentazione, interamente in inglese, è divisa in:

- `README.md`: contiene una breve descrizione dello scopo del progetto, come impostarlo, come usarlo e brevi descrizioni per ciascuno degli algoritmi implementati nei risolutori.
- `CHANGELOG.md`: contiene la cronologia delle versioni e dei cambiamenti apportati.
- Sphinx Doc: l'intero codice sorgente è commentato tramite Docstring, le quali vengono poi usate, tramite Sphinx, per costruire documentazione ben formattata che viene poi resa disponibile online all'indirizzo https://rchimisso.github.io/mcs-prog-1/.

### CI

La libreria è hostata su GitHub e sono presenti workflow personalizzati per il rilascio di nuove versioni.

Quando viene effettuato un rilascio:

1. È necessario specificare il nuovo numero di versione, il quale sarà validato per assicurarsi che rispetti il formato corretto e che non sia una regressione.
2. Il codice sorgente viene analizzato tramite Prospector per assicurarsi di mantenere lo stile di scrittura coerente e pulito.
3. Il codice sorgente viene testato tramite i test presenti al fine di evitare regressioni sulle funzionalità.
4. Se tutti i passi precedenti hanno esito positivo, il numero di versione viene aggiornato.
5. Eseguibili per Linux e Windows vengono generati e una nuova release viene creata.
6. Viene costruita la documentazione aggiornata tramite Sphinx e viene pubblicata con GitHub Pages all'indirizzo https://rchimisso.github.io/mcs-prog-1/.

## Risultati

TODO

## Conclusioni

TODO
