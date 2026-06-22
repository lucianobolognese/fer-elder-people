# FER Elder People: Emotion Recognition

> Tesi di Laurea in Informatica — Sistemi ad Agenti  
> Università degli Studi di Bari Aldo Moro, A.A. 2025–2026  
> **Laureando:** Luciano Domenico Bolognese  
> **Relatori:** Prof.ssa Berardina Nadja De Carolis · Dr. Giuseppe Palestra

---

## Panoramica

Questo repository contiene il codice, la pipeline di costruzione del dataset e gli esperimenti sviluppati nell'ambito della tesi *"Analisi comparativa dei metodi di riconoscimento delle emozioni primarie dal volto negli anziani"*.

Il lavoro affronta un problema sistematico e poco esplorato nel campo del **Facial Expression Recognition (FER)**: i modelli dello stato dell'arte degradano significativamente quando applicati a volti di soggetti anziani. Questo fenomeno — l'**age bias** — non è riconducibile a limitazioni architetturali, ma alla distribuzione demografica dei dataset su cui i modelli vengono addestrati.

Il contributo principale è **SAGE-Face**, un dataset costruito ex novo composto interamente da volti di soggetti anziani, accompagnato da un benchmark comparativo su cinque framework FER e da un processo di fine-tuning che dimostra come il bias sia correggibile.

---

## Contenuto della repository

```
├── dataset/
│   ├── pipeline/           # Script di costruzione SAGE-Face
│   │   ├── age_screening.py        # Stima età con DeepFace
│   │   ├── vlm_validation.py       # Validazione con Qwen3-VL
│   │   └── merge_sources.py        # Aggregazione sorgenti statiche/dinamiche
│   └── sage_face_master.csv        # Master CSV con Unique_ID, path, Ground Truth
│
├── benchmark/
│   ├── pipeline_inference.py   # Pipeline di inferenza unificata per tutti i modelli
│   ├── label_normalization.py  # Mappatura etichette → 7 emozioni di Ekman
│   └── results/
│       ├── accuracy_table.csv
│       ├── precision_table.csv
│       └── f1_table.csv
│
├── gradcam/
│   ├── gradcam_analysis.py     # Estrazione heatmap con hook PyTorch
│   └── visualize.py            # Overlay JET colormap su immagini originali
│
├── finetuning/
│   ├── dataset_builder.py      # Aggregazione dataset per fine-tuning
│   ├── augmentation.py         # Pipeline Albumentations + Specific Erasing
│   ├── train.py                # Loop di addestramento AdamW + CosineAnnealingLR
│   ├── cross_validation.py     # 5-Fold Cross Validation
│   └── eval.py                 # Valutazione su tutti i dataset post fine-tuning
│
└── notebooks/
    ├── benchmark_analysis.ipynb    # Analisi e visualizzazione risultati benchmark
    └── finetuning_results.ipynb    # Curve di training, confusion matrix, Grad-CAM
```

---

## Il problema: age bias nel FER

I modelli FER contemporanei raggiungono accuratezze elevate sui propri dataset di addestramento, ma falliscono sistematicamente quando applicati a soggetti anziani. La causa è morfologica e precisa:

- **Rughe statiche** → attivano le stesse feature convoluzionali associate a contrazioni muscolari emotive (es. AU12, AU17)
- **Ptosi palpebrale** → riduce la leggibilità della regione oculare
- **Solchi nasolabiali profondi** → generano falsi positivi verso le classi *Angry* e *Sad*
- **Cedimento dei tessuti molli** → altera la geometria facciale attesa dai modelli

Il fenomeno è dimostrato quantitativamente dal benchmark: i modelli specializzati su RAF-DB perdono oltre **40 punti percentuali di F1-Score** quando valutati su SAGE-Face.

---

## Dataset

### Dataset usati nel benchmark

| Dataset | Immagini | Risoluzione | Note |
|---------|----------|-------------|------|
| AffectNet | 30.626 | 96×96 | In-the-wild, annotazione singolo esperto |
| RAF-DB | 15.339 | 100×100 | Crowdsourcing 40+ annotatori |
| FER-2013 | 35.887 | 48×48 | Grayscale, benchmark Kaggle 2013 |
| RAISE-FER | 224.028 | 48×48 | Augmented (17 trasformazioni), bilanciato |
| **SAGE-Face** | **9.566** | **224×224** | **Anziani — costruito in questa tesi** |

### SAGE-Face

SAGE-Face è ottenuto aggregando sei sorgenti eterogenee:

**Statiche:** AffectNet · FACES · RAF-DB · FERPlus  
**Dinamiche (frame estratti da video):** DFEW · ElderReact

La pipeline di costruzione è articolata in tre stadi:

1. **Screening demografico** — DeepFace stima l'età per ogni immagine; vengono conservati solo i soggetti con età stimata ≥ 60 anni.
2. **Validazione multimodale** — Qwen3-VL analizza ogni frame con un prompt di ispezione binario (YES/NO) verificando la presenza di marcatori fisiologici dell'invecchiamento (perdita di elasticità epidermica, rughe statiche, identificatori senili). I dataset clinicamente certificati come anziani (FACES, ElderReact) bypassano questo stadio.
3. **Normalizzazione etichette** — le label eterogenee vengono mappate alle 7 emozioni discrete di Ekman (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).

**Split finale:** 7.650 train · 1.916 test (split ufficiale, mai alterato)

> ⚠️ Il test set non va mai soggetto a data augmentation. È costruito per rappresentare una condizione reale in-the-wild e serve da specchio fedele per la valutazione comparativa.

---

## Framework analizzati

| Framework | Paradigma | Note |
|-----------|-----------|------|
| OpenFace 3.0 | Landmark + AU tracking | CE-CLM, analisi geometrica esplicita |
| RMN | Deep Residual + Masking | Attenzione spaziale su regioni facciali |
| DDAMFN | Dual Attention | Attenzione parziale + canale |
| EfficientFace | Lightweight CNN | Feature locali + modulatori spaziali |
| **EmotiEffLib** | **Multi-Task Learning** | **EfficientNet-B2 + valenza/arousal/AU — baseline più robusta** |

Tutti i modelli sono valutati in modalità **zero-shot** sui test set ufficiali. Nessun fine-tuning sui dataset target per preservare l'autenticità del benchmark.

---

## Risultati benchmark (F1-Score macro)

| Modello | Pesi | AffectNet | RAF-DB | FER-2013 | RAISE-FER | SAGE-Face |
|---------|------|-----------|--------|----------|-----------|-----------|
| EmotiEffLib | AffectNet | **0.686** | 0.618 | 0.520 | 0.506 | **0.528** |
| DDAMFN | AffectNet | 0.650 | — | — | — | 0.528 |
| DDAMFN | RAF-DB | — | 0.742 | 0.504 | 0.488 | 0.478 |
| EfficientFace | AffectNet | 0.600 | — | — | 0.448 | 0.485 |
| EfficientFace | RAF-DB | — | **0.822** | 0.439 | 0.410 | 0.411 |
| RMN | FER-2013 | 0.306 | 0.314 | **0.669** | **0.664** | 0.442 |
| OpenFace 3.0 | Base | 0.525 | 0.410 | 0.390 | 0.390 | 0.383 |

**Osservazione chiave:** nessun modello supera F1 0.53 su SAGE-Face. Il calo non è proporzionale alla complessità del modello — EfficientFace (RAF-DB), tra le architetture più avanzate, degrada più di OpenFace. Il bias è demografico, non architetturale.

---

## Analisi Grad-CAM

Per identificare *dove* il modello concentra l'attenzione durante la classificazione, è stata applicata la tecnica **Gradient-weighted Class Activation Mapping** al modello EmotiEffLib.

I gradienti vengono registrati tramite hook PyTorch agganciati ai blocchi convoluzionali `features_extractor.blocks.4` e `features_extractor.blocks.5` — scelta motivata dal compromesso ottimale tra profondità semantica e risoluzione spaziale.

**Risultato:** su soggetti anziani con Ground Truth *Neutral* classificati erroneamente come *Angry*, la rete attiva intensamente la regione nasolabiale e perioculare inferiore — zone dove le rughe statiche sono morfologicamente più marcate. Le stesse regioni che in un soggetto giovane indicherebbero AU12 (sorriso) o AU17 (abbassamento del mento). Il modello non distingue la deformazione senile del tessuto dal movimento muscolare intenzionale.

---

## Fine-Tuning su SAGE-Face

### Approccio

Transfer learning da EmotiEffLib (pesi AffectNet) su un dataset aggregato:

- **7.650** immagini da SAGE-Face train set
- **1.000** immagini da RAF-DB, AffectNet e RAISE-FER (campionamento bilanciato per classe)

L'inclusione di dati non anziani è una scelta metodologica deliberata: evitare il *catastrophic forgetting* delle capacità di generalizzazione acquisite nel pre-addestramento.

### Data augmentation (solo train set)

Pipeline Albumentations con 5 versioni aumentate per immagine: flip orizzontale, rotazioni stocastiche (±15°), distorsioni di griglia, variazioni di luminosità/contrasto, blur gaussiano, rumore gaussiano. Inoltre **Specific Erasing** — occlusione selettiva della regione oculare o labiale nel 30% dei campioni, tramite classificatori OpenCV, per aumentare la robustezza a occlusioni parziali.

### Iperparametri

| Parametro | Valore |
|-----------|--------|
| Ottimizzatore | AdamW |
| Learning rate | 1×10⁻⁵ |
| Batch size | 16 |
| Epoche massime | 150 |
| Early stopping patience | 12 epoche |
| LR scheduler | CosineAnnealingLR |
| Hardware | NVIDIA RTX 5060, 32GB RAM |

Il training si è concluso all'**epoca 70** per intervento dell'early stopping.

### Risultati

| Metodo | Pesi | AffectNet | RAF-DB | FER-2013 | RAISE-FER | SAGE-Face |
|--------|------|-----------|--------|----------|-----------|-----------|
| EmotiEffLib | AffectNet (zero-shot) | 0.686 | 0.618 | 0.520 | 0.506 | 0.528 |
| EmotiEffLib | **Fine-Tuning** | **0.735** | **0.733** | **0.569** | **0.572** | **0.731** |
| | Δ | +4.9pp | +11.5pp | +4.9pp | +6.6pp | **+20.3pp** |

Il fine-tuning su soggetti anziani non produce alcuna regressione sugli altri dataset — anzi, migliora le prestazioni ovunque. La variabilità morfologica dei soggetti anziani ha operato come **regolarizzazione implicita**.

Gli errori *Neutral→Angry* — il pattern di misclassificazione più ricorrente — si riducono dell'**83%** dopo il fine-tuning.

### 5-Fold Cross Validation

| Fold | Accuracy | F1-Score macro |
|------|----------|----------------|
| 1 | 74.72% | 0.7318 |
| 2 | 75.06% | 0.7374 |
| 3 | 74.90% | 0.7361 |
| 4 | 75.35% | 0.7396 |
| 5 | 75.30% | 0.7400 |
| **Media ± σ** | **75.07% ± 0.24%** | **0.7370 ± 0.003** |

La deviazione standard di **0.003** conferma che F1 0.731 sul test set ufficiale non è il risultato di uno split fortunato. La bassa varianza tra fold è anche un indicatore indiretto della qualità distributiva di SAGE-Face.

---

## Requisiti

```bash
pip install torch torchvision
pip install deepface
pip install albumentations
pip install scikit-learn
pip install opencv-python
pip install emotiefflib        # EmotiEffLib / HSEmotion
pip install openface           # OpenFace 3.0
```

Per la validazione VLM è necessario Qwen3-VL in esecuzione locale.

---

## Citazione

Se utilizzi SAGE-Face o il codice di questo repository, cita:

```bibtex
@thesis{bolognese2026sageface,
  author  = {Bolognese, Luciano Domenico},
  title   = {Analisi comparativa dei metodi di riconoscimento delle emozioni primarie dal volto negli anziani},
  school  = {Università degli Studi di Bari Aldo Moro},
  year    = {2026},
  type    = {Tesi di Laurea Triennale},
  advisor = {De Carolis, Berardina Nadja and Palestra, Giuseppe}
}
```

---

## Licenza

Il codice è rilasciato sotto licenza MIT. SAGE-Face è distribuito esclusivamente per fini di ricerca accademica, nel rispetto delle licenze dei dataset sorgente (AffectNet, RAF-DB, FER-2013, RAISE-FER, DFEW, ElderReact, FACES, FERPlus).
