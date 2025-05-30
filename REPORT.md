# Relazione

##### *2025, Metodi del Calcolo Scientifico, Riccardo Chimisso 866009 & Mauro Zorzin 866001*

## Indice

1. [Struttura](#struttura)
    1. [Linguaggio](#linguaggio)
    2. [Codice](#codice)
        1. [`solvers.py`](#solverspy)
        2. [`main.py`](#mainpy)
    3. [Implementazioni](#implementazioni)
        1. [Jacobi](#jacobi)
        2. [Gauss-Seidel](#gauss-seidel)
        3. [Gradient Descent](#gradient-descent)
        4. [Conjugate Gradient](#conjugate-gradient)
    4. [Dati](#dati)
    5. [Test](#test)
    6. [Documentazione](#documentazione)
    7. [CI](#ci)
2. [Risultati](#risultati)
    1. [Tempo per iterazione](#tempo-per-iterazione)
    2. [Grafici iterazioni impiegate](#grafici-iterazioni-impiegate)
    3. [Grafici tempo per tolleranza](#grafici-tempo-per-tolleranza)
3. [Conclusioni](#conclusioni)
  1. [Informazioni sulle matrici](#informazioni-sulle-matrici)
  2. [Commenti](#commenti)
4. [Sviluppi futuri](#sviluppi-futuri)
5. [Appendice](#appendice)
    1. [Dati grezzi](#dati-grezzi)
    2. [Dati tabellari](#dati-tabellari)

## Struttura

*La struttura di questa libreria è molto semplice e descritta di seguito.*  
*Una descrizione più dettagliata di ciascun modulo, classe e metodo è sempre disponibile per l'ultima release all'indirizzo https://rchimisso.github.io/mcs-prog-1/.*

### Linguaggio

Il linguaggio utilizzato è Python, la cui scelta è stata motivata dalla sua semplicità implementativa, ampia diffusione e supporto esteso da parte della comunità scientifica.  
In particolare, le numerose librerie scientifiche già disponibili, come NumPy e SciPy, permettono di sfruttare strutture dati efficienti e operazioni matematiche ottimizzate, facilitando notevolmente l'implementazione e la verifica degli algoritmi.

### Codice

Il codice sorgente di questa libreria è in Python ed è diviso in due moduli principali: `solver.py` e `main.py`.

#### `solvers.py`

Questo è il modulo che contiene il codice dei diversi risolutori.

La classe astratta `IterativeSolver` fornisce un'interfaccia comune.  
Questo permette di riutilizzare codice, sia internamente, come per il calcolo dell'errore residuo, sia esternamente, nel caso si voglia astrarre la logica applicativa dall'implementazione dello specifico risolutore.

Grazie alla classe astratta, ciascun risolutore può essere inizializzato con valori per tolleranza (`tol`) e numero massimo di iterazioni (`max_iter`) di default. I valori passati durante l'istanziazione del risolutore verranno usati qualora non vengano forniti nella chiamata al metodo `solve`.

Il metodo `solve` richiede di passare la matrice $A$ e il vettore $b$.  
Opzionalmente è possibile specificare una soluzione di partenza $x_0$ diversa dal vettore con tutti $0$, così come una tolleranza o un numero massimo di iterazioni diversi da quelli di base del risolutore.  
Il risultato restituito è una tupla con la soluzione approssimata trovata e i dati raccolti sull'esecuzione.  
Il metodo si arresta quando raggiunge il numero massimo di iterazioni consentite oppure ottiene una soluzione il cui errore residuo $\frac{|| Ax - b ||}{|| b ||}$ è inferiore alla tolleranza specificata.

I 4 risolutori disponibili sono classi concrete che ereditano da `IterativeSolver`: `JacobiSolver`, `GaussSeidelSolver`, `GradientDescentSolver` e `ConjugateGradientSolver`.  
Ciascun risolutore ha nel nome l'algoritmo risolutivo che implementa.

È inoltre presente la *dataclass* `Info` che racchiude i dati per un'esecuzione di un risolutore, permettendo di disaccoppiare i risultati degli algoritmi da quelli di stampa.

È importante sottolineare che tutta la matematica gira in `float64` e i metodi presuppongono matrici SPD (simmetriche e definite positive).

#### `main.py`

Questo è il modulo principale per l'utilizzo della libreria a linea di comando.

Contiene la logica per il parsing degli argomenti a linea di comando, la lettura di file `.mtx`, l'esecuzione dei diversi risolutori coi parametri specificati e la stampa dei relativi risultati.

### Implementazioni

#### Jacobi

```python
def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
  # Punto di partenza per la ricerca della soluzione.
  x = x0
  # Estrai la diagonale di A.
  D = np.diag(A)
  # Estrai la matrice "remainder" di A.
  R = A - np.diagflat(D)
  # Itera al massimo max_iter volte e tieni traccia del progresso.
  for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
    # Aggiorna la soluzione approssimata.
    x = (b - R @ x) / D
    # Controlla l'errore residuo e confrontalo con la tolleranza.
    if self._residual(A, b, x) < tol:
      return x, k + 1, True
  return x, max_iter, False
```

#### Gauss-Seidel

```python
def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
  # Punto di partenza per la ricerca della soluzione e sua dimensione.
  x, n = x0, b.size
  # Estrai la diagonale di A, il triangolo inferiore e il triangolo superiore.
  D, L, U = np.diag(A), np.tril(A, -1), np.triu(A,  1)
  # Itera al massimo max_iter volte e tieni traccia del progresso.
  for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
    # Esegui la forward substitution per aggiornare la soluzione approssimata.
    for i in range(n):
      x[i] = (b[i] - L[i] @ x - U[i] @ x) / D[i]
    # Controlla l'errore residuo e confrontalo con la tolleranza.
    if self._residual(A, b, x) < tol:
      return x, k + 1, True
  return x, max_iter, False
```

#### Gradient Descent

```python
def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
  # Punto di partenza per la ricerca della soluzione e residuo iniziale.
  x, r = x0, b - A @ x0
  # Verifica se il residuo iniziale soddisfa già il criterio di arresto.
  if np.linalg.norm(r) > tol * np.linalg.norm(b):
    # Itera al massimo max_iter volte e tieni traccia del progresso.
    for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
      # Calcola la direzione del gradiente.
      Ar = A @ r
      # Calcola il passo ottimale di discesa (Steepest Descent).
      alpha = (r @ r) / (r @ Ar)
      # Aggiorna la soluzione approssimata.
      x = x + alpha * r
      # Aggiorna il residuo associato alla nuova soluzione approssimata.
      r = r - alpha * Ar
      # Controlla l'errore residuo e confrontalo con la tolleranza.
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
    return x, max_iter, False
  return x, 0, True
```

#### Conjugate Gradient

```python
def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
  # Punto di partenza per la ricerca della soluzione e residuo iniziale.
  x, r = x0, b - A @ x0
  # Direzione coniugata e valore del residuo scalare.
  p, rs_old = r.copy(), r @ r
  # Verifica se il residuo iniziale soddisfa già il criterio di arresto.
  if np.linalg.norm(r) > tol * np.linalg.norm(b):
    # Itera al massimo max_iter volte e tieni traccia del progresso.
    for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
      # Calcola la direzione coniugata.
      Ap = A @ p
      # Coefficiente di spostamento lungo la direzione coniugata.
      alpha = rs_old / (p @ Ap)
      # Aggiorna la soluzione approssimata.
      x = x + alpha * p
      # Aggiorna il residuo associato alla nuova soluzione approssimata.
      r = r - alpha * Ap
      # Controlla l'errore residuo e confrontalo con la tolleranza.
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
      # Aggiorna il residuo scalare.
      rs_new = r @ r
      # Coefficiente di ricorrenza per rendere p coniugata rispetto ad A.
      beta = rs_new / rs_old
      # Aggiorna la direzione coniugata di spostamento.
      p = r + beta * p
      # Aggiorna il vecchio residuo scalare per l'iterazione successiva.
      rs_old = rs_new
    return x, max_iter, False
  return x, 0, True
```

### Dati

All'interno della cartella [data](./data/) sono presenti alcuni file `.mtx` di base come esempio da poter usare per confrontare i diversi risolutori.

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

I workflow di cui ai punti 2, 3 e 6 sono eseguibili anche in modo indipendente.

## Risultati

Sono stati fatti $16$ esperimenti per ogni combinazione di matrici di test (`spa1.mtx`, `spa2.mtx`, `vem1.mtx`, `vem2.mtx`) e tolleranza ($1\mathrm{e}{-4}$, $1\mathrm{e}{-6}$, $1\mathrm{e}{-8}$, $1\mathrm{e}{-10}$).  
Il numero massimo di iterazioni è stato impostato a $20000$.  
Ogni esperimento è stato eseguito $5$ volte e di seguito sono riportati i risultati migliori, in quanto le misure possono essere affette da altri programmi che girano sulla stessa macchina, e quindi il tempo più accurato è il più basso (anziché la media).

Tutti gli esperimenti sono stati eseguiti sul medesimo computer con le seguenti specifiche:

- CPU: 12th Gen Intel(R) Core(TM) i7-12700H 2.30 GHz
- RAM: 64 GB
- Archiviazione: 1 TB SSD Samsung SSD 980 PRO 1TB
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB)
- OS: Windows 10 Home 22H2 (build 19045.5854)

### Tempo per iterazione

Per ciascuna coppia matrice-risolutore è riportato il tempo in secondi impiegato per ciascuna iterazione.

|        | Jacobi              | Gauss-Seidel        | Gradient Descent  | Conjugate Gradient |
|--------|:-------------------:|:-------------------:|:-----------------:|:------------------:|
| `spa1` | $3\mathrm{e}{-4}$   | $4.5\mathrm{e}{-3}$ | $3\mathrm{e}{-4}$ | $3\mathrm{e}{-4}$  |
| `spa2` | $3\mathrm{e}{-3}$   | $2\mathrm{e}{-2}$   | $3\mathrm{e}{-3}$ | $3\mathrm{e}{-3}$  |
| `vem1` | $1\mathrm{e}{-3}$   | $8\mathrm{e}{-3}$   | $8\mathrm{e}{-4}$ | $8\mathrm{e}{-4}$  |
| `vem2` | $2.5\mathrm{e}{-3}$ | $1.4\mathrm{e}{-2}$ | $2\mathrm{e}{-3}$ | $2\mathrm{e}{-3}$  |

### Grafici iterazioni impiegate

I grafici sotto riportati illustrano, per ogni risolutore, il numero di iterazioni impiegate al variare del valore di tolleranza e matrice.

![spa1-iter](/assets/spa1-iter.png)
![spa2-iter](/assets/spa2-iter.png)
![vem1-iter](/assets/vem1-iter.png)
![vem2-iter](/assets/vem2-iter.png)

### Grafici tempo per tolleranza

I grafici sotto riportati illustrano, per ogni risolutore, l'andamento di tempo impiegato al variare di tolleranza e matrice.  
I grafici sono in scala semilogaritmica (asse x) e le ascisse sono invertite di segno.

![spa1-time](/assets/spa1-time.png)
![spa2-time](/assets/spa2-time.png)
![vem1-time](/assets/vem1-time.png)
![vem2-time](/assets/vem2-time.png)

## Conclusioni

### Informazioni sulle matrici

| Sigla  | n      | nnz         | Densità | Diagonalmente Dominante | $\lambda_{min}$ | $\lambda_{max}$ | κ           |
| ------ | ------ | ----------- | ------- | :---------------------: | --------------: | --------------: | ----------: |
| `spa1` | $1000$ | $≈1.8*10^5$ | $18\%$  | no                      | $≈0.488$        | $≈1000$         | $≈2*10^3$   |
| `spa2` | $3000$ | $≈1.6*10^6$ | $18\%$  | no                      | $≈2.124$        | $≈3000$         | $≈1.5*10^3$ |
| `vem1` | $1681$ | $≈1.3*10^4$ | $0.5\%$ | no                      | $≈0.012$        | $≈4$            | $≈3*10^2$   |
| `vem2` | $2601$ | $≈2.1*10^4$ | $0.3\%$ | no                      | $≈0.008$        | $≈4$            | $≈5*10^2$   |

*Nota:* $κ ≃ \frac{\lambda_{max}}{\lambda_{min}}$ *numero di condizionamento.*

### Commenti

Tutti i metodi arrivano a convergenza entro il limite di $20000$ iterazioni, tuttavia i risultati ottenuti evidenziano con chiarezza quanto la scelta dell'algoritmo di risoluzione incida sia sul tempo complessivo sia sul numero di iterazioni richieste.  
L'analisi evidenzia anche che l'efficacia di ciascun metodo dipende fortemente dalla struttura e dal condizionamento del sistema lineare.

Per matrici più dense e peggio condizionate (*spa1*, *spa2*), Gauss–Seidel (*GS*) primeggia nel contenimento del numero di iterazioni, seguito a breve distanza da Conjugate Gradient (*CG*) e da Jacobi (*J*), mentre Gradient Descent (GD) si conferma largamente inadeguato.  
Sui sistemi più sparsi e meglio condizionati (*vem1*, *vem2*) la situazione si ribalta: *CG* diviene nettamente il più efficiente, con *GS* secondo, GD terzo e *J* relegato all'ultima posizione.

In termini di tempo di esecuzione complessivo, GD risulta sempre il più lento; *GS*, pur garantendo un minor numero di iterazioni rispetto a *J*, paga un costo per iterazione più elevato e la natura intrinsecamente sequenziale lo rendono globalmente meno competitivo, infatti supera Jacobi solo in casi particolarmente gravosi (come *spa2*) dove la riduzione delle iterazioni compensa il maggior overhead.  
*CG* rimane una scelta di gran lunga preferibile rispetto a GD, grazie al drastico risparmio di iterazioni a parità di costo per iterazione, nonché la più rapida in assoluto, salvo per *spa2* dove i metodi stazionari risultano competitivi.

In conclusione, *GS* è indicato solo per matrici molto grandi e dense, *CG* rappresenta la soluzione ottimale nella maggior parte degli scenari, GD va evitato, mentre *J* resta utile principalmente in contesti didattici o fortemente parallelizzati, dove la semplicità implementativa e la possibilità di parallelismo possono compensare la lenta velocità di convergenza.

## Sviluppi futuri

Possibili direzioni di sviluppi futuri:

- Permettere la scelta della precisione, ad esempio `float32` e `float16`. Attualmente tutte le computazioni girano in `float64`.
- Integrare versioni GPU per testare il vantaggio di Jacobi su architetture massivamente parallele.
- Permettere più esecuzioni senza dover rilanciare lo script.
- Automatizzare l'analisi delle matrici in input, la creazione di grafici e l'esportazione dei dati.

## Appendice

Sotto sono riportati i dati per ciascun esperimento, sia in forma grezza che in forma tabella.

### Dati grezzi

**Input**  
&emsp;Matrice: `spa1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-4}$  
**Jacobi Solver**  
&emsp;Rel. error: $0.001771281148305271$  
&emsp;Iterations: $115$  
&emsp;Iterations/s: $3258.60$  
&emsp;Time elapsed: $0.05057860002852976$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $0.01820594299518869$  
&emsp;Iterations: $9$  
&emsp;Iterations/s: $195.05$  
&emsp;Time elapsed: $0.05004250002093613$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $0.034574700773010954$  
&emsp;Iterations: $143$  
&emsp;Iterations/s: $3302.22$  
&emsp;Time elapsed: $0.04630160005763173$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $0.02078976000995675$  
&emsp;Iterations: $49$  
&emsp;Iterations/s: $3204.20$  
&emsp;Time elapsed: $0.019374200026504695$ s  

**Input**  
&emsp;Matrice: `spa1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-6}$  
**Jacobi Solver**  
&emsp;Rel. error: $1.7979295435203664\mathrm{e}{-05}$  
&emsp;Iterations: $181$  
&emsp;Iterations/s: $3333.27$  
&emsp;Time elapsed: $0.07098550000227988$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $0.00012996939585935833$  
&emsp;Iterations: $17$  
&emsp;Iterations/s: $219.13$  
&emsp;Time elapsed: $0.08329209999646991$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $0.0009680457310610868$  
&emsp;Iterations: $3577$  
&emsp;Iterations/s: $3035.66$  
&emsp;Time elapsed: $1.1822739000199363$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $2.5529092775890698\mathrm{e}{-05}$  
&emsp;Iterations: $134$  
&emsp;Iterations/s: $2463.03$  
&emsp;Time elapsed: $0.05895330000203103$ s  

**Input**  
&emsp;Matrice: `spa1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-8}$  
**Jacobi Solver**  
&emsp;Rel. error: $1.8249788574507168\mathrm{e}{-07}$  
&emsp;Iterations: $247$  
&emsp;Iterations/s: $3074.89$  
&emsp;Time elapsed: $0.09630870004184544$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $1.709732900192307\mathrm{e}{-06}$  
&emsp;Iterations: $24$  
&emsp;Iterations/s: $217.01$  
&emsp;Time elapsed: $0.11760290001984686$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $9.816363743987784\mathrm{e}{-06}$  
&emsp;Iterations: $8233$  
&emsp;Iterations/s: $3025.04$  
&emsp;Time elapsed: $2.7249416999984533$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $1.3198404441969156\mathrm{e}{-07}$  
&emsp;Iterations: $177$  
&emsp;Iterations/s: $2885.35$  
&emsp;Time elapsed: $0.06593149993568659$ s  

**Input**  
&emsp;Matrice: `spa1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-10}$  
**Jacobi Solver**  
&emsp;Rel. error: $1.8524352742197325\mathrm{e}{-09}$  
&emsp;Iterations: $313$  
&emsp;Iterations/s: $3056.20$  
&emsp;Time elapsed: $0.11736450006719679$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $2.248087991277066\mathrm{e}{-08}$  
&emsp;Iterations: $31$  
&emsp;Iterations/s: $222.37$  
&emsp;Time elapsed: $0.14502769999671727$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $9.82038624643504\mathrm{e}{-08}$  
&emsp;Iterations: $12919$  
&emsp;Iterations/s: $2853.14$  
&emsp;Time elapsed: $4.530833700089715$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $1.2031105079603335\mathrm{e}{-09}$  
&emsp;Iterations: $200$  
&emsp;Iterations/s: $2927.66$  
&emsp;Time elapsed: $0.07181039999704808$ s  

**Input**  
&emsp;Matrice: `spa2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-4}$  
**Jacobi Solver**  
&emsp;Rel. error: $0.001766246519114273$  
&emsp;Iterations: $36$  
&emsp;Iterations/s: $296.47$  
&emsp;Time elapsed: $0.19387769990134984$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $0.00259889558745254$  
&emsp;Iterations: $5$  
&emsp;Iterations/s: $41.67$  
&emsp;Time elapsed: $0.17381860001478344$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $0.018129645117112236$  
&emsp;Iterations: $161$  
&emsp;Iterations/s: $272.11$  
&emsp;Time elapsed: $0.6105576999252662$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $0.00982112845726422$  
&emsp;Iterations: $42$  
&emsp;Iterations/s: $327.98$  
&emsp;Time elapsed: $0.16092469997238368$ s  

**Input**  
&emsp;Matrice: `spa2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-6}$  
**Jacobi Solver**  
&emsp;Rel. error: $1.6667561367165373\mathrm{e}{-05}$  
&emsp;Iterations: $57$  
&emsp;Iterations/s: $302.73$  
&emsp;Time elapsed: $0.26648700004443526$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $5.141641259558678\mathrm{e}{-05}$  
&emsp;Iterations: $8$  
&emsp;Iterations/s: $48.31$  
&emsp;Time elapsed: $0.22096010006498545$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $0.0006694229253854699$  
&emsp;Iterations: $1949$  
&emsp;Iterations/s: $311.26$  
&emsp;Time elapsed: $6.281709000002593$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $0.00011979846178536083$  
&emsp;Iterations: $122$  
&emsp;Iterations/s: $315.10$  
&emsp;Time elapsed: $0.41637320001609623$ s  

**Input**  
&emsp;Matrice: `spa2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-8}$  
**Jacobi Solver**  
&emsp;Rel. error: $1.5728698925534617\mathrm{e}{-07}$  
&emsp;Iterations: $78$  
&emsp;Iterations/s: $315.70$  
&emsp;Time elapsed: $0.33538249996490777$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $2.7943220320511893\mathrm{e}{-07}$  
&emsp;Iterations: $12$  
&emsp;Iterations/s: $50.43$  
&emsp;Time elapsed: $0.2970233999658376$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $6.8652401165109194\mathrm{e}{-06}$  
&emsp;Iterations: $5087$  
&emsp;Iterations/s: $324.11$  
&emsp;Time elapsed: $15.714612699928693$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $5.586660602468392\mathrm{e}{-07}$  
&emsp;Iterations: $196$  
&emsp;Iterations/s: $333.90$  
&emsp;Time elapsed: $0.6159428999526426$ s  

**Input**  
&emsp;Matrice: `spa2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-10}$  
**Jacobi Solver**  
&emsp;Rel. error: $1.4842728068778481\mathrm{e}{-09}$  
&emsp;Iterations: $99$  
&emsp;Iterations/s: $307.20$  
&emsp;Time elapsed: $0.40341119992081076$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $5.57073907340775\mathrm{e}{-09}$  
&emsp;Iterations: $15$  
&emsp;Iterations/s: $49.51$  
&emsp;Time elapsed: $0.3636420000111684$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $6.937814076870232\mathrm{e}{-08}$  
&emsp;Iterations: $8285$  
&emsp;Iterations/s: $325.49$  
&emsp;Time elapsed: $25.473189599928446$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $5.32423004396093\mathrm{e}{-09}$  
&emsp;Iterations: $240$  
&emsp;Iterations/s: $327.12$  
&emsp;Time elapsed: $0.7618696999270469$ s  

**Input**  
&emsp;Matrice: `vem1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-4}$  
**Jacobi Solver**  
&emsp;Rel. error: $0.0035403807574087773$  
&emsp;Iterations: $1314$  
&emsp;Iterations/s: $843.51$  
&emsp;Time elapsed: $1.5889921999769285$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $0.0035069725970372618$  
&emsp;Iterations: $659$  
&emsp;Iterations/s: $125.71$  
&emsp;Time elapsed: $5.260676400037482$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $0.0027045724093676133$  
&emsp;Iterations: $890$  
&emsp;Iterations/s: $1186.92$  
&emsp;Time elapsed: $0.7571939000627026$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $4.08279315868558\mathrm{e}{-05}$  
&emsp;Iterations: $38$  
&emsp;Iterations/s: $506.84$  
&emsp;Time elapsed: $0.08409240003675222$ s  

**Input**  
&emsp;Matrice: `vem1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-6}$  
**Jacobi Solver**  
&emsp;Rel. error: $3.5400733430273\mathrm{e}{-05}$  
&emsp;Iterations: $2433$  
&emsp;Iterations/s: $972.59$  
&emsp;Time elapsed: $2.529653000063263$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $3.526696849439208\mathrm{e}{-05}$  
&emsp;Iterations: $1218$  
&emsp;Iterations/s: $127.23$  
&emsp;Time elapsed: $9.594218100071885$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $2.7133391835656714\mathrm{e}{-05}$  
&emsp;Iterations: $1612$  
&emsp;Iterations/s: $1275.69$  
&emsp;Time elapsed: $1.2712017999729142$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $3.732339701260148\mathrm{e}{-07}$  
&emsp;Iterations: $45$  
&emsp;Iterations/s: $1157.63$  
&emsp;Time elapsed: $0.049089399981312454$ s  

**Input**  
&emsp;Matrice: `vem1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-8}$  
**Jacobi Solver**  
&emsp;Rel. error: $3.539765957961423\mathrm{e}{-07}$  
&emsp;Iterations: $3552$  
&emsp;Iterations/s: $985.75$  
&emsp;Time elapsed: $3.6336801999714226$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $3.5174569999343266\mathrm{e}{-07}$  
&emsp;Iterations: $1778$  
&emsp;Iterations/s: $127.15$  
&emsp;Time elapsed: $14.007255800068378$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $2.6953373951515354\mathrm{e}{-07}$  
&emsp;Iterations: $2336$  
&emsp;Iterations/s: $1189.32$  
&emsp;Time elapsed: $1.970805300050415$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $2.8318733771496425\mathrm{e}{-09}$  
&emsp;Iterations: $53$  
&emsp;Iterations/s: $1189.46$  
&emsp;Time elapsed: $0.05535229993984103$ s  

**Input**  
&emsp;Matrice: `vem1.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-10}$  
**Jacobi Solver**  
&emsp;Rel. error: $3.539458862537178\mathrm{e}{-09}$  
&emsp;Iterations: $4671$  
&emsp;Iterations/s: $949.11$  
&emsp;Time elapsed: $4.950419400003739$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $3.5082426887404108\mathrm{e}{-09}$  
&emsp;Iterations: $2338$  
&emsp;Iterations/s: $125.27$  
&emsp;Time elapsed: $18.686488199979067$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $2.7131674485379716\mathrm{e}{-09}$  
&emsp;Iterations: $3058$  
&emsp;Iterations/s: $1246.84$  
&emsp;Time elapsed: $2.4588767999084666$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $2.191749791757509\mathrm{e}{-11}$  
&emsp;Iterations: $59$  
&emsp;Iterations/s: $1149.69$  
&emsp;Time elapsed: $0.06041809997987002$ s  

**Input**  
&emsp;Matrice: `vem2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-4}$  
**Jacobi Solver**  
&emsp;Rel. error: $0.004968461406187168$  
&emsp;Iterations: $1927$  
&emsp;Iterations/s: $400.65$  
&emsp;Time elapsed: $4.866523200063966$ s   
**Gauss-Seidel Solver**  
&emsp;Rel. error: $0.004951189291537078$  
&emsp;Iterations: $965$  
&emsp;Iterations/s: $71.99$  
&emsp;Time elapsed: $13.451204799930565$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $0.0038119295293834498$  
&emsp;Iterations: $1308$  
&emsp;Iterations/s: $426.36$  
&emsp;Time elapsed: $3.0824144000653177$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $5.729017222223839\mathrm{e}{-05}$  
&emsp;Iterations: $47$  
&emsp;Iterations/s: $446.65$  
&emsp;Time elapsed: $0.12505149992648512$ s  

**Input**  
&emsp;Matrice: `vem2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-6}$  
**Jacobi Solver**  
&emsp;Rel. error: $4.967034430530644\mathrm{e}{-05}$  
&emsp;Iterations: $3676$  
&emsp;Iterations/s: $417.21$  
&emsp;Time elapsed: $8.86670100002084$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $4.941761266875986\mathrm{e}{-05}$  
&emsp;Iterations: $1840$  
&emsp;Iterations/s: $70.09$  
&emsp;Time elapsed: $26.29788800003007$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $3.7914189772545046\mathrm{e}{-05}$  
&emsp;Iterations: $2438$  
&emsp;Iterations/s: $437.81$  
&emsp;Time elapsed: $5.583444000105374$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $4.7429962822323075\mathrm{e}{-07}$  
&emsp;Iterations: $56$  
&emsp;Iterations/s: $410.44$  
&emsp;Time elapsed: $0.15555390005465597$ s  

**Input**  
&emsp;Matrice: `vem2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-8}$  
**Jacobi Solver**  
&emsp;Rel. error: $4.965607735630084\mathrm{e}{-07}$  
&emsp;Iterations: $5425$  
&emsp;Iterations/s: $383.72$  
&emsp;Time elapsed: $14.19685889990069$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $4.958369542525802\mathrm{e}{-07}$  
&emsp;Iterations: $2714$  
&emsp;Iterations/s: $70.35$  
&emsp;Time elapsed: $38.62406530010048$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $3.8098505016615633\mathrm{e}{-07}$  
&emsp;Iterations: $3566$  
&emsp;Iterations/s: $466.12$  
&emsp;Time elapsed: $7.665427000029013$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $4.29998333776711\mathrm{e}{-09}$  
&emsp;Iterations: $66$  
&emsp;Iterations/s: $454.60$  
&emsp;Time elapsed: $0.16625719994772226$ s  

**Input**  
&emsp;Matrice: `vem2.mtx`  
&emsp;Tolleranza: $1\mathrm{e}{-10}$  
**Jacobi Solver**  
&emsp;Rel. error: $4.964168598820624\mathrm{e}{-09}$  
&emsp;Iterations: $7174$  
&emsp;Iterations/s: $414.10$  
&emsp;Time elapsed: $17.37905230000615$ s  
**Gauss-Seidel Solver**  
&emsp;Rel. error: $4.948898161655902\mathrm{e}{-09}$  
&emsp;Iterations: $3589$  
&emsp;Iterations/s: $70.83$  
&emsp;Time elapsed: $50.72326670004986$ s  
**Gradient Descent Solver**  
&emsp;Rel. error: $3.7987565270044575\mathrm{e}{-09}$  
&emsp;Iterations: $4696$  
&emsp;Iterations/s: $459.04$  
&emsp;Time elapsed: $10.24550039996393$ s  
**Conjugate Gradient Solver**  
&emsp;Rel. error: $2.247621137824687\mathrm{e}{-11}$  
&emsp;Iterations: $74$  
&emsp;Iterations/s: $439.76$  
&emsp;Time elapsed: $0.1893309000879526$ s  

### Dati tabellari

| Matrice    | Tolleranza         | Risolutore         | Errore relativo                     | Iterazioni | Iterazioni al secondo | Tempo impiegato (s)   |
|------------|:------------------:|--------------------|:------------------------------------|-----------:|----------------------:|----------------------:|
| `spa1.mtx` | $1\mathrm{e}{-4}$  | Jacobi             | $0.0017712811483052$                | $115$      | $3258.60$             | $0.0505786000285297$  |
| `spa1.mtx` | $1\mathrm{e}{-4}$  | Gauss-Seidel       | $0.0182059429951886$                | $9$        | $195.05$              | $0.0500425000209361$  |
| `spa1.mtx` | $1\mathrm{e}{-4}$  | Gradient Descent   | $0.0345747007730109$                | $143$      | $3302.22$             | $0.0463016000576317$  |
| `spa1.mtx` | $1\mathrm{e}{-4}$  | Conjugate Gradient | $0.0207897600099567$                | $49$       | $3204.20$             | $0.0193742000265046$  |
| `spa1.mtx` | $1\mathrm{e}{-6}$  | Jacobi             | $1.7979295435203664\mathrm{e}{-05}$ | $181$      | $3333.27$             | $0.0709855000022798$  |
| `spa1.mtx` | $1\mathrm{e}{-6}$  | Gauss-Seidel       | $0.0001299693958593$                | $17$       | $219.13$              | $0.0832920999964699$  |
| `spa1.mtx` | $1\mathrm{e}{-6}$  | Gradient Descent   | $0.000968045731061$                 | $3577$     | $3035.66$             | $1.1822739000199365$  |
| `spa1.mtx` | $1\mathrm{e}{-6}$  | Conjugate Gradient | $2.5529092775890695\mathrm{e}{-05}$ | $134$      | $2463.03$             | $0.0589533000020310$  |
| `spa1.mtx` | $1\mathrm{e}{-8}$  | Jacobi             | $1.824978857450717\mathrm{e}{-07}$  | $247$      | $3074.89$             | $0.0963087000418454$  |
| `spa1.mtx` | $1\mathrm{e}{-8}$  | Gauss-Seidel       | $1.709732900192307\mathrm{e}{-06}$  | $24$       | $217.01$              | $0.1176029000198468$  |
| `spa1.mtx` | $1\mathrm{e}{-8}$  | Gradient Descent   | $9.816363743987784\mathrm{e}{-06}$  | $8233$     | $3025.04$             | $2.7249416999984533$  |
| `spa1.mtx` | $1\mathrm{e}{-8}$  | Conjugate Gradient | $1.3198404441969156\mathrm{e}{-07}$ | $177$      | $2885.35$             | $0.0659314999356865$  |
| `spa1.mtx` | $1\mathrm{e}{-10}$ | Jacobi             | $1.8524352742197323\mathrm{e}{-09}$ | $313$      | $3056.20$             | $0.1173645000671967$  |
| `spa1.mtx` | $1\mathrm{e}{-10}$ | Gauss-Seidel       | $2.248087991277066\mathrm{e}{-08}$  | $31$       | $222.37$              | $0.1450276999967172$  |
| `spa1.mtx` | $1\mathrm{e}{-10}$ | Gradient Descent   | $9.82038624643504\mathrm{e}{-08}$   | $12919$    | $2853.14$             | $4.5308337000897150$  |
| `spa1.mtx` | $1\mathrm{e}{-10}$ | Conjugate Gradient | $1.2031105079603335\mathrm{e}{-09}$ | $200$      | $2927.66$             | $0.0718103999970480$  |
| `spa2.mtx` | $1\mathrm{e}{-4}$  | Jacobi             | $0.0017662465191142$                | $36$       | $296.47$              | $0.1938776999013498$  |
| `spa2.mtx` | $1\mathrm{e}{-4}$  | Gauss-Seidel       | $0.0025988955874525$                | $5$        | $41.67$               | $0.1738186000147834$  |
| `spa2.mtx` | $1\mathrm{e}{-4}$  | Gradient Descent   | $0.0181296451171122$                | $161$      | $272.11$              | $0.6105576999252662$  |
| `spa2.mtx` | $1\mathrm{e}{-4}$  | Conjugate Gradient | $0.0098211284572642$                | $42$       | $327.98$              | $0.1609246999723836$  |
| `spa2.mtx` | $1\mathrm{e}{-6}$  | Jacobi             | $1.6667561367165373\mathrm{e}{-05}$ | $57$       | $302.73$              | $0.2664870000444352$  |
| `spa2.mtx` | $1\mathrm{e}{-6}$  | Gauss-Seidel       | $5.141641259558678\mathrm{e}{-05}$  | $8$        | $48.31$               | $0.2209601000649854$  |
| `spa2.mtx` | $1\mathrm{e}{-6}$  | Gradient Descent   | $0.0006694229253854$                | $1949$     | $311.26$              | $6.2817090000025930$  |
| `spa2.mtx` | $1\mathrm{e}{-6}$  | Conjugate Gradient | $0.0001197984617853$                | $122$      | $315.1$               | $0.4163732000160962$  |
| `spa2.mtx` | $1\mathrm{e}{-8}$  | Jacobi             | $1.5728698925534617\mathrm{e}{-07}$ | $78$       | $315.70$              | $0.3353824999649077$  |
| `spa2.mtx` | $1\mathrm{e}{-8}$  | Gauss-Seidel       | $2.7943220320511893\mathrm{e}{-07}$ | $12$       | $50.43$               | $0.2970233999658376$  |
| `spa2.mtx` | $1\mathrm{e}{-8}$  | Gradient Descent   | $6.86524011651092\mathrm{e}{-06}$   | $5087$     | $324.11$              | $15.7146126999286920$ |
| `spa2.mtx` | $1\mathrm{e}{-8}$  | Conjugate Gradient | $5.586660602468392\mathrm{e}{-07}$  | $196$      | $333.90$              | $0.6159428999526426$  |
| `spa2.mtx` | $1\mathrm{e}{-10}$ | Jacobi             | $1.484272806877848\mathrm{e}{-09}$  | $99$       | $307.20$              | $0.4034111999208107$  |
| `spa2.mtx` | $1\mathrm{e}{-10}$ | Gauss-Seidel       | $5.570739073407751\mathrm{e}{-09}$  | $15$       | $49.51$               | $0.3636420000111684$  |
| `spa2.mtx` | $1\mathrm{e}{-10}$ | Gradient Descent   | $6.937814076870232\mathrm{e}{-08}$  | $8285$     | $325.49$              | $25.4731895999284500$ |
| `spa2.mtx` | $1\mathrm{e}{-10}$ | Conjugate Gradient | $5.3242300439609305\mathrm{e}{-09}$ | $240$      | $327.12$              | $0.7618696999270469$  |
| `vem1.mtx` | $1\mathrm{e}{-4}$  | Jacobi             | $0.0035403807574087$                | $1314$     | $843.51$              | $1.5889921999769283$  |
| `vem1.mtx` | $1\mathrm{e}{-4}$  | Gauss-Seidel       | $0.0035069725970372$                | $659$      | $125.71$              | $5.2606764000374820$  |
| `vem1.mtx` | $1\mathrm{e}{-4}$  | Gradient Descent   | $0.0027045724093676$                | $890$      | $1186.92$             | $0.7571939000627026$  |
| `vem1.mtx` | $1\mathrm{e}{-4}$  | Conjugate Gradient | $4.08279315868558\mathrm{e}{-05}$   | $38$       | $506.84$              | $0.0840924000367522$  |
| `vem1.mtx` | $1\mathrm{e}{-6}$  | Jacobi             | $3.5400733430273\mathrm{e}{-05}$    | $2433$     | $972.59$              | $2.5296530000632630$  |
| `vem1.mtx` | $1\mathrm{e}{-6}$  | Gauss-Seidel       | $3.526696849439208\mathrm{e}{-05}$  | $1218$     | $127.23$              | $9.5942181000718850$  |
| `vem1.mtx` | $1\mathrm{e}{-6}$  | Gradient Descent   | $2.7133391835656718\mathrm{e}{-05}$ | $1612$     | $1275.69$             | $1.2712017999729142$  |
| `vem1.mtx` | $1\mathrm{e}{-6}$  | Conjugate Gradient | $3.732339701260148\mathrm{e}{-07}$  | $45$       | $1157.63$             | $0.0490893999813124$  |
| `vem1.mtx` | $1\mathrm{e}{-8}$  | Jacobi             | $3.539765957961423\mathrm{e}{-07}$  | $3552$     | $985.75$              | $3.6336801999714230$  |
| `vem1.mtx` | $1\mathrm{e}{-8}$  | Gauss-Seidel       | $3.5174569999343266\mathrm{e}{-07}$ | $1778$     | $127.15$              | $14.0072558000683780$ |
| `vem1.mtx` | $1\mathrm{e}{-8}$  | Gradient Descent   | $2.695337395151536\mathrm{e}{-07}$  | $2336$     | $1189.32$             | $1.9708053000504150$  |
| `vem1.mtx` | $1\mathrm{e}{-8}$  | Conjugate Gradient | $2.831873377149642\mathrm{e}{-09}$  | $53$       | $1189.46$             | $0.0553522999398410$  |
| `vem1.mtx` | $1\mathrm{e}{-10}$ | Jacobi             | $3.539458862537178\mathrm{e}{-09}$  | $4671$     | $949.11$              | $4.9504194000037390$  |
| `vem1.mtx` | $1\mathrm{e}{-10}$ | Gauss-Seidel       | $3.5082426887404104\mathrm{e}{-09}$ | $2338$     | $125.27$              | $18.6864881999790670$ |
| `vem1.mtx` | $1\mathrm{e}{-10}$ | Gradient Descent   | $2.7131674485379716\mathrm{e}{-09}$ | $3058$     | $1246.84$             | $2.4588767999084660$  |
| `vem1.mtx` | $1\mathrm{e}{-10}$ | Conjugate Gradient | $2.1917497917575087\mathrm{e}{-11}$ | $59$       | $1149.69$             | $0.0604180999798700$  |
| `vem2.mtx` | $1\mathrm{e}{-4}$  | Jacobi             | $0.0049684614061871$                | $1927$     | $400.65$              | $4.8665232000639660$  |
| `vem2.mtx` | $1\mathrm{e}{-4}$  | Gauss-Seidel       | $0.004951189291537$                 | $965$      | $71.99$               | $13.4512047999305630$ |
| `vem2.mtx` | $1\mathrm{e}{-4}$  | Gradient Descent   | $0.0038119295293834$                | $1308$     | $426.36$              | $3.0824144000653177$  |
| `vem2.mtx` | $1\mathrm{e}{-4}$  | Conjugate Gradient | $5.729017222223839\mathrm{e}{-05}$  | $47$       | $446.65$              | $0.1250514999264851$  |
| `vem2.mtx` | $1\mathrm{e}{-6}$  | Jacobi             | $4.967034430530644\mathrm{e}{-05}$  | $3676$     | $417.21$              | $8.8667010000208400$  |
| `vem2.mtx` | $1\mathrm{e}{-6}$  | Gauss-Seidel       | $4.941761266875986\mathrm{e}{-05}$  | $1840$     | $70.09$               | $26.2978880000300700$ |
| `vem2.mtx` | $1\mathrm{e}{-6}$  | Gradient Descent   | $3.7914189772545046\mathrm{e}{-05}$ | $2438$     | $437.81$              | $5.5834440001053740$  |
| `vem2.mtx` | $1\mathrm{e}{-6}$  | Conjugate Gradient | $4.7429962822323085\mathrm{e}{-07}$ | $56$       | $410.44$              | $0.1555539000546559$  |
| `vem2.mtx` | $1\mathrm{e}{-8}$  | Jacobi             | $4.965607735630084\mathrm{e}{-07}$  | $5425$     | $383.72$              | $14.1968588999006900$ |
| `vem2.mtx` | $1\mathrm{e}{-8}$  | Gauss-Seidel       | $4.958369542525802\mathrm{e}{-07}$  | $2714$     | $70.35$               | $38.6240653001004800$ |
| `vem2.mtx` | $1\mathrm{e}{-8}$  | Gradient Descent   | $3.8098505016615633\mathrm{e}{-07}$ | $3566$     | $466.12$              | $7.6654270000290130$  |
| `vem2.mtx` | $1\mathrm{e}{-8}$  | Conjugate Gradient | $4.2999833377671106\mathrm{e}{-09}$ | $66$       | $454.60$              | $0.1662571999477222$  |
| `vem2.mtx` | $1\mathrm{e}{-10}$ | Jacobi             | $4.964168598820624\mathrm{e}{-09}$  | $7174$     | $414.10$              | $17.3790523000061500$ |
| `vem2.mtx` | $1\mathrm{e}{-10}$ | Gauss-Seidel       | $4.948898161655902\mathrm{e}{-09}$  | $3589$     | $70.83$               | $50.7232667000498600$ |
| `vem2.mtx` | $1\mathrm{e}{-10}$ | Gradient Descent   | $3.7987565270044575\mathrm{e}{-09}$ | $4696$     | $459.04$              | $10.2455003999639300$ |
| `vem2.mtx` | $1\mathrm{e}{-10}$ | Conjugate Gradient | $2.247621137824687\mathrm{e}{-11}$  | $74$       | $439.76$              | $0.1893309000879526$  |
