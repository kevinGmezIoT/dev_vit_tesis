# Visual Feature Extractor (Person Re-ID) — Embedding promedio por secuencia (MARS)

Este repositorio implementa **el módulo de extracción de características visuales (apariencia)** para re-identificación de personas (video-based Re-ID).  
El módulo recibe como entrada una **secuencia de imágenes RGB recortadas** (frames del cuerpo/ROI de la persona) y devuelve un **embedding único por secuencia**, calculado como el **promedio** de embeddings por frame + **normalización L2**.

> Alcance: este repo se centra solo en el *visual feature extractor*. La tesis completa contempla también gait, fusión multimodal y destilación; aquí lo dejamos listo para integrarse más adelante.

---

## 1) Objetivos del proyecto

1. Entrenar un extractor visual robusto (Vision Transformer) para generar un embedding discriminativo por identidad.
2. Consumir el dataset **MARS** como conjunto de entrenamiento/evaluación basado en **tracklets/secuencias**.
3. Proveer:
   - `train.py`: entrenamiento supervisado con pérdidas típicas de Re-ID.
   - `inference.py`: inferencia simple para obtener el embedding de una secuencia.
   - `config.yaml`: hiperparámetros reproducibles.
   - Pipeline de preprocesamiento para convertir tracklets a secuencias de longitud fija.

---

## 2) Arquitectura del modelo (propuesta)

### 2.1. Backbone (por frame): ViT
- Backbone sugerido: `ViT-Base/16` (opcional: `ViT-Small/16` si requieres eficiencia).
- Entrada típica: `256x128` o `384x192`.
- Cabezal Re-ID:
  - Proyección a dimensión `d` (ej. 512 o 768).
  - Normalización `L2` del embedding.

### 2.2. Agregación temporal (por secuencia): Average Pooling
Dada una secuencia de `T` frames ya recortados:
1. Se calcula un embedding por frame: `e_t = f_vit(x_t)`  (dimensión `d`)
2. Se obtiene el embedding por secuencia (tracklet/ciclo):
   - `e_seq = mean_t(e_t)`
3. Se aplica normalización:
   - `e_seq = e_seq / ||e_seq||_2`

> Esto cumple tu requisito: “recibir secuencia RGB y obtener embedding promedio”.

---

## 3) Dataset: MARS (estructura esperada en este repo)

MARS se distribuye como videos/tracklets; este proyecto asume que cada tracklet ya corresponde a una identidad y cámara (y, si deseas, a un “ciclo”/ventana temporal).  
Para entrenamiento estable, convertimos cada tracklet a **secuencias de longitud fija**:

- `sequence_len`: 25 frames (por defecto)
- `overlap`: 10 frames (por defecto)

### 3.1. Estructura final procesada (recomendada)

Después del preprocesamiento, se recomienda la siguiente estructura:

```
data/
  mars/
    raw/                # MARS original (tal como lo descargues)
    processed/
      train/
        ID0001/
          cam01_seq0001/
            rgb/
              frame0001.jpg
              frame0002.jpg
              ...
          cam02_seq0007/
            rgb/...
        ID0002/
          ...
      val/
        ...
      test/
        ...
```

> Nota: En MARS normalmente se trabaja con split train/test oficial. Aquí añadimos `val/` opcional (sub-split del train) para tuning.

### 3.2. Archivo índice (recomendado)
Para trazabilidad y entrenamiento eficiente, genera un CSV (o parquet) con:
- `split` (train/val/test)
- `person_id`
- `camera_id`
- `sequence_id`
- `frames_dir` (ruta a la carpeta `rgb/`)
- `num_frames`

Ejemplo: `data/mars/processed/index.csv`

---

## 4) Estructura del repositorio

```
.
├── config.yaml
├── train.py
├── inference.py
├── data/
│   └── mars/
│       ├── raw/
│       └── processed/
├── preprocess/
│   ├── prepare_mars.py          # convierte tracklets a secuencias fijas + index.csv
│   ├── transforms.py            # transforms/augmentations (torchvision)
│   └── dataset.py               # Dataset / DataLoader (secuencias)
├── model/
│   ├── vit_embedder.py          # ViT backbone + proyección + L2 norm
│   ├── temporal_pooling.py      # average pooling por secuencia
│   └── losses.py                # CE (label smoothing), Triplet (batch-hard)
├── notebooks/
│   ├── 01_exploracion_mars.ipynb
│   └── 02_sanity_check_embeddings.ipynb
└── README.md
```

---

## 5) Instalación del entorno

### 5.1. Requisitos
- Python 3.10+
- GPU NVIDIA (recomendado)

### 5.2. Crear entorno e instalar dependencias
Ejemplo con `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm opencv-python pandas pyyaml tqdm scikit-learn
```

---

## 6) Preprocesamiento de MARS

1) Coloca el dataset original en:
```
data/mars/raw/
```

2) Ejecuta el script de preparación:

```bash
python preprocess/prepare_mars.py \
  --raw_root data/mars/raw \
  --out_root data/mars/processed \
  --sequence_len 25 \
  --overlap 10
```

Salida:
- Carpetas `train/`, `val/` (opcional), `test/`
- `data/mars/processed/index.csv`

---

## 7) Entrenamiento

`train.py` entrena el extractor visual supervisado por identidad.

### 7.1. Pérdidas recomendadas (clásico Re-ID)
- `L_CE`: Cross-Entropy / Identity loss (con label smoothing)
- `L_triplet`: Triplet loss (batch-hard) usando muestreo **P×K**

Pérdida total:
```
L = λ_id * L_CE + λ_tri * L_triplet
```

> En este repo omitimos (por ahora) coherencia contextual y destilación para mantener el alcance “solo visual”.

### 7.2. Ejecutar entrenamiento
```bash
python train.py --config config.yaml
```

Artefactos esperados:
- `runs/<exp_name>/checkpoints/best.pth`
- `runs/<exp_name>/config_resolved.yaml`
- logs de métricas (loss, acc top-1, etc.)

---

## 8) Inferencia (embedding de una secuencia)

### 8.1. Inferencia desde un directorio de frames
Estructura esperada:
```
mi_secuencia/
  frame0001.jpg
  frame0002.jpg
  ...
```

Ejecuta:
```bash
python inference.py \
  --config config.yaml \
  --checkpoint runs/exp/checkpoints/best.pth \
  --frames_dir mi_secuencia \
  --output embedding.npy
```

Salida:
- `embedding.npy` con shape `[d]`.

### 8.2. Uso en un sistema Re-ID (idea)
- Obtén `e_seq_query` (consulta) y `e_seq_gallery` (galería).
- Calcula similitud coseno: `sim = dot(e1, e2)` (ya están L2-normalizados).
- Rankea por similitud.

---

## 9) Configuración (`config.yaml`)

Ejemplo mínimo (ajústalo a tu GPU/dataset):

```yaml
experiment:
  name: vit_mars_seqavg

data:
  root: data/mars/processed
  index_csv: data/mars/processed/index.csv
  image_size: [256, 128]     # [H, W]
  sequence_len: 25
  num_workers: 4

model:
  backbone: vit_base_patch16_224   # timm name (puedes usar vit_small_patch16_224)
  pretrained: true
  embed_dim: 512
  dropout: 0.1

train:
  epochs: 60
  batch:
    P: 16          # identidades por batch
    K: 4           # secuencias por identidad
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.05
  scheduler:
    name: cosine
    warmup_epochs: 5

loss:
  ce:
    label_smoothing: 0.1
    weight: 1.0
  triplet:
    margin: 0.3
    weight: 1.0

eval:
  normalize_embeddings: true
```

---

## 10) Detalles de implementación (qué debe codificar un LLM)

### 10.1. `preprocess/dataset.py`
- Debe cargar una secuencia:
  - Lee `sequence_len` imágenes desde `frames_dir` (si faltan, muestrea/replica).
  - Devuelve tensor: `x` con shape `[T, 3, H, W]` y label `person_id`.

### 10.2. `model/vit_embedder.py`
- Usa `timm.create_model(backbone, pretrained=...)`
- Obtén representación por frame (ej. token CLS o pooled features).
- Cabezal:
  - Linear -> `embed_dim`
  - `F.normalize(embedding, dim=-1)` para L2.
- Clasificador:
  - Linear -> `num_ids` (solo para entrenamiento).

### 10.3. `model/temporal_pooling.py`
- Recibe embeddings por frame `[B, T, d]` y retorna `[B, d]`:
  - `e_seq = mean(dim=1)`
  - `e_seq = normalize(e_seq)`

### 10.4. `model/losses.py`
- `CrossEntropyLoss` con label smoothing.
- `BatchHardTripletLoss` (min pos/max neg por ancla dentro del batch P×K).

### 10.5. `train.py`
- Construye dataloaders con muestreo P×K.
- Forward:
  - frames -> embeddings por frame -> mean -> embedding secuencia
  - logits para CE
- Backprop y logging.

### 10.6. `inference.py`
- Carga checkpoint
- Lee todos los frames del directorio
- Aplica transforms
- Produce embedding promedio + L2 y lo guarda.

---

## 11) Checklist rápido

- [ ] `data/mars/raw` poblado con MARS
- [ ] `python preprocess/prepare_mars.py ...` genera `processed/` + `index.csv`
- [ ] `python train.py --config config.yaml` entrena y guarda `best.pth`
- [ ] `python inference.py ...` exporta `embedding.npy`

---

## 12) Próximos pasos (fuera del alcance del módulo visual)
- Integrar embeddings espacio-temporales y/o gait.
- Añadir destilación Maestro–Estudiante si se requiere inferencia RGB-only con robustez transferida.
- Añadir métricas Re-ID estándar (CMC, mAP) sobre el protocolo oficial de MARS.
