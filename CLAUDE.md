# dagn_lib — Desarrollo y contexto

## Identidad del proyecto

**dagn_lib** es la contribución doctoral principal del sistema DAGN.
A diferencia de dagn_simple (que usa encoders CNN/LSTM pesados, 8.8M params),
dagn_lib extrae **features bibliográficamente fundamentadas** con librerías
establecidas (MediaPipe, NeuroKit2, MNE) y las fusiona con un LSTM minimalista
(81K params). La defensa doctoral se centra en esta elección arquitectónica.

```
face(T,17) + physio(T,6) + eeg(T,5) → LayerNorm(28) → LSTM(128) → Linear(2) → tanh → VA
```

Total: **81,210 parámetros** — defendible en 5 minutos en una pizarra.

---

## Estado actual (2026-03-17)

### Modelo — ENTRENAMIENTO FINALIZADO
- Checkpoint: `production/fusion_best.pth`
- Best CCC (val): **0.326** — epoch 127, early stopping en epoch 167 (PATIENCE=40)
- Arquitectura: `FusionLSTM` en `production/fusion_model.py`

### Evaluación final (split=val, N=937)

| Dataset | N | CCC-V | 95% CI | CCC-A | 95% CI | Media |
|---------|---|-------|--------|-------|--------|-------|
| DEAP | 256 | 0.155 | [0.030, 0.268] | 0.080 | [-0.036, 0.194] | 0.117 |
| WESAD | 299 | 0.550 | [0.466, 0.617] | 0.547 | [0.458, 0.626] | **0.549** |
| DREAMER | 82 | -0.156 | [-0.356, 0.042] | 0.129 | [-0.078, 0.342] | -0.014 |
| AFEW-VA | 182 | 0.514 | [0.406, 0.600] | 0.455 | [0.331, 0.551] | **0.484** |
| AFFEC | 118 | 0.312 | [0.145, 0.468] | 0.142 | [-0.037, 0.314] | 0.227 |
| **GLOBAL** | **937** | **0.329** | **[0.272, 0.386]** | **0.309** | **[0.247, 0.370]** | **0.319** |

**Tabla LaTeX lista en `results_log.txt`.**

Análisis por dataset:
- **WESAD 0.549** — señales fisiológicas de muñeca (BVP+EDA+TEMP) capturan arousal fisiológico bien
- **AFEW-VA 0.484** — AUs MediaPipe geométricos funcionan para vídeo facial
- **AFFEC 0.227** — AUs OpenFace2 + EEG + GSR; arousal débil (0.142), valence moderada
- **DEAP 0.117** — EEG 32ch→4Hz muy submuestreado; mayor pérdida de información
- **DREAMER valence -0.156** — conocido en literatura (escala Likert con poca varianza)

Comparativa con dagn_simple (8.8M params, evaluación en conjunto completo):
- dagn_lib (81K): WESAD=0.549, AFEW-VA=0.484, GLOBAL=0.319
- dagn_simple (8.8M): WESAD=0.598, AFEW-VA=0.372, GLOBAL=0.435
- dagn_lib supera en AFEW-VA (AUs vs CNN features), pero GLOBAL menor por DREAMER/DEAP

**Argumento doctoral**: 100× menos parámetros, features bibliográficas explícitas, AFEW-VA comparable.

### Producción desplegada
- Servicio FastAPI: `production/analizar_emocion_service.py` ✅
- Dashboard Streamlit: `production/dashboard.py` ✅
- Script arranque dashboard: `production/start_dashboard.sh` ✅

---

## Arquitectura de features

### face (T=30, dim=17) — Action Units geométricos
Extraídos con **MediaPipe FaceMesh** (468 landmarks, `feature_extractor_face.py`).
Referencia: Ekman & Friesen (1978) FACS.

| Índice | AU | Descripción |
|--------|-----|-------------|
| 0 | AU01 | Inner Brow Raise |
| 1 | AU02 | Outer Brow Raise |
| 2 | AU04 | Brow Lowerer |
| 3 | AU06 | Cheek Raiser |
| 4 | AU07 | Lid Tightener |
| 5 | AU10 | Upper Lip Raiser |
| 6 | AU12 | Lip Corner Puller |
| 7 | AU14 | Dimpler |
| 8 | AU15 | Lip Corner Depressor |
| 9 | AU17 | Chin Raiser |
| 10 | AU20 | Lip Stretcher |
| 11 | AU23 | Lip Tightener |
| 12 | AU24 | Lip Pressor |
| 13 | AU25 | Lips Part |
| 14 | AU26 | Jaw Drop |
| 15 | AU28 | Lip Suck |
| 16 | AU43 | Eyes Closed (EAR) |

En AFFEC se usan los AU_r reales de OpenFace2 directamente.
Zeros cuando no hay vídeo (DEAP, WESAD, DREAMER).

### physio (T=30, dim=6) — HRV + EDA + TEMP
Extraídos con **NeuroKit2** (`feature_extractor_physio.py`).
Referencias: Task Force ESC/NASPE (1996), Boucsein (2012).

| Col | Feature | Fórmula/normalización |
|-----|---------|----------------------|
| 0 | RMSSD_norm | RMSSD(ms) / 100 |
| 1 | SDNN_norm | SDNN(ms) / 100 |
| 2 | mean_HR_norm | HR(BPM) / 100 |
| 3 | EDA_tonic | SCL, z-scored (NeuroKit2 EDA decomp) |
| 4 | EDA_phasic | SCR, z-scored |
| 5 | TEMP_norm | (°C - 34) / 4 |

HRV (cols 0-2): estadísticos de ventana → replicados en los T timesteps.
EDA y TEMP (cols 3-5): por timestep via mean-pooling.

### eeg (T=30, dim=5) — Bandpower + Asimetría frontal
Extraídos con **MNE-Python** (`feature_extractor_eeg.py`), fallback numpy.
Referencias: Davidson (1988), Klimesch (1999).

| Col | Feature | Descripción |
|-----|---------|-------------|
| 0 | theta_log | log(theta 4-8Hz), z-scored |
| 1 | alpha_log | log(alpha 8-13Hz), z-scored |
| 2 | beta_log | log(beta 13-30Hz), z-scored |
| 3 | alpha_asym | (R-L)/(R+L+ε), frontal, Davidson 1988 |
| 4 | theta_alpha | log(θ/α ratio), Klimesch 1999 |

En producción (NeuroSky): theta≈log1p((1-att)*0.5), alpha≈log1p(med*0.5),
beta≈log1p(att*0.5), asym=0.0, ratio=theta-alpha.

---

## Datasets de entrenamiento

| Dataset | Modalidades usadas | Muestras | Ruta |
|---------|-------------------|----------|------|
| DEAP | EEG 32ch (MNE) + BVP/GSR/TEMP (NK2) | 1280 | `/mnt/f/source_datasets/Fisiologico/DEAP/` |
| WESAD | BVP/EDA/TEMP wrist (NK2) | 1498 | `/home/alvar/datasets/WESAD` ⚠️ BORRAR tras entrenar |
| DREAMER | EEG 14ch (MNE) + ECG (NK2) | 414 | `/mnt/f/source_datasets/Fisiologico/DREAMER/` |
| AFEW-VA | AUs MediaPipe (pre-extraídos) | 914 | `/mnt/f/source_datasets/Fisiologico/AFEW-VA/` |
| AFFEC | AUs OpenFace2 + EEG + GSR | 594 | `/home/alvar/datasets/affec_features/` |

### Features pre-extraídas (NO repetir)
- AFEW-VA AUs: `~/datasets/afew_va_au_features/` — 914 .npy (457 clips + 457 flipped)
  - Script: `training/extract_afew_au_features.py`
- AFFEC: `~/datasets/affec_features/` — 594 .npz
  - Script: `training/extract_affec_features.py`

### Dataset bias — solución aplicada
Cada dataset usa escala VA distinta (1-9, 1-5, 0-10).
Solución: **z-score por sub-dataset** antes de entrenar (`global_dataset.py`).
→ El modelo aprende dinámica intra-dataset, no offsets inter-dataset.

---

## Modelo FusionLSTM

```python
# production/fusion_model.py
x = concat(face, physio, eeg)      # (B, T, 28)
x = LayerNorm(28)(x)               # normaliza escala entre modalidades
h, _ = LSTM(28→128, layers=1)(x)   # aprende dinámica temporal multimodal
h = Dropout(0.3)(h)
va = tanh(Linear(128→2)(h))        # (B, T, 2) valence/arousal ∈ [-1, 1]
grad = diff(va, dim=T)             # gradiente temporal (para producción)
```

**Parámetros**: LayerNorm(56) + LSTM(79,872) + Linear(258) = **80,186**

---

## Training (`training/train_fusion.py`)

```
Loss = MSE + (1 - CCC_valence) + (1 - CCC_arousal) + variance_penalty
Optimizer: AdamW, LR=1e-3, WD=1e-4
Scheduler: CosineAnnealingLR
Early stopping: PATIENCE=40
Modal dropout: P_face=0.2, P_physio=0.2, P_eeg=0.3
Variance penalty alphas: DEAP=0.5, WESAD=0.0, DREAMER=0.5, AFEW-VA=0.3, AFFEC=0.3
EPOCHS=200, BATCH_SIZE=32, T=30
```

Checkpoint guardado en `production/fusion_best.pth` cuando mejora CCC de validación.

---

## Producción (sistema tiempo real)

### Arquitectura del sistema
```
Sensores ESP32 (MQTT 50Hz)
    │ topic: tesis/biomedidas
    │ campos: ir, red, gsr, tmp  (sensor fisiológico)
    │         att, med            (sensor EEG NeuroSky)
    ▼
Dashboard Streamlit  ──── HTTP POST ──► FastAPI :8000 /analyze
    │    (dashboard.py)                  (analizar_emocion_service.py)
    │                                          │
    │    Recibe raw values:              MediaPipe AUs (desde disco)
    │    ir, gsr, temp, att, med         NeuroKit2 HRV/EDA/TEMP
    │                                    EEG approx (att/med → features)
    │                                    FusionLSTM → VA + grad
    │                                          │
    └── Muestra: VA plane, VA timeline,  hr_bpm, rmssd, eda_mean ◄─┘
        raw IR/GSR/ATT/MED charts
        cámara MJPEG (Flask Windows)
```

### Servicio (`production/analizar_emocion_service.py`)
- **Request**: `{session_id, ir, red, gsr, temp, att, med, ir_batch[], red_batch[], gsr_batch[], temp_batch[]}`
  - `ir_batch` etc: todos los samples MQTT acumulados desde la última llamada (50 Hz real)
  - Si no hay batch, usa el valor único `ir/gsr/temp` (fallback)
- **Session reset**: cuando `session_id` cambia, limpia todos los deques y EMA
- **Physio**: NeuroKit2 sobre bvp_deque (50 Hz, cap 15s), zeros si < 5s
- **SpO2**: ratio-of-ratios IR/RED últimos 5s → `SpO2 = 110 - 25*R`
- **rPPG**: green channel forehead ROI → Welch PSD → HR cam (0.7–min(3.0,nyq×0.9) Hz) y resp (0.1–0.5 Hz); requiere `nyq > 0.8`
- **Blinks**: `ear_hires` buffer a framerate de cámara (máx 8 extracciones AU43/llamada)
- **EMA smoothing**: alpha=0.25 sobre salida VA cruda (evita saturación tanh)
- **Face-only forward**: pasa physio/eeg=zeros para obtener `face_v`/`face_a` (indicador solo cámara)
- **Warmup**: T=30 llamadas antes de predicciones (~30s a 1000ms/refresh)
- **Response**: `{status, valence, arousal, grad_v, grad_a, face_v, face_a, hr_bpm, rmssd, eda_mean, spo2, cam_hr, cam_resp, blink_rate, warmup}`

### Dashboard (`production/dashboard.py`)
- **Batch physio**: acumula MQTT desde `last_physio_ts` → envía `ir_batch[]` al servicio (50 Hz real)
- **`last_physio_ts`**: se inicializa a `datetime.now()` al START SESSION (evita seed histórico)
- **Autorefresh**: 1000ms
- **Layout**: `st.columns([4,1])` — izquierda: cámara+VA plane+gráficas; derecha: métricas fijas
  - Subcols cámara | plano VA (arriba), VA timeline + raw signals (abajo, sin scroll)
  - Métricas divididas en secciones **Sensor** y **Camera**
- **VA plane**: punto naranja "Cam" (face_v/face_a), leyenda en `y=-0.25`
- **VA timeline**: `key="va_timeline"`, `width=2`, `connectgaps=True`, `range=[-1.05,1.05]`

### Campos MQTT reales (tesis/biomedidas)
```
Sensor fisiológico: ir, red, gsr, tmp    (50 Hz)
Sensor EEG:         att, med, sensor, signal
```
Los dos sensores publican mensajes separados → df tiene NaN en columnas del otro sensor.
Separar con: `df[df["ir"].notna()]` y `df[df["att"].notna()]`.

### Cámara
- Windows Flask server en `http://172.26.96.1:5000`
- `/video_feed` → MJPEG stream directo en el dashboard
- `/record_start?session_id=X`, `/record_stop` → guarda frames en `/mnt/c/biometria_tesis/X/`
- El servicio lee frames desde `/mnt/c/biometria_tesis/{session_id}/*.jpg`

---

## Comandos de arranque

```bash
# 1. Mosquitto (permite conexión ESP32)
mosquitto -v -c /etc/mosquitto/mosquitto.conf > /tmp/mosquitto.log 2>&1 &

# 2. FastAPI service
cd /home/alvar/dagn_lib/production
/home/alvar/venv_tesis/bin/uvicorn analizar_emocion_service:app --host 0.0.0.0 --port 8000 > /tmp/service.log 2>&1 &

# 3. Dashboard — USAR SIEMPRE setsid para que no muera al cerrar el shell
setsid bash /home/alvar/dagn_lib/production/start_dashboard.sh > /tmp/dashboard.log 2>&1 &
# Token CAM_API_TOKEN viene de production/.streamlit/secrets.toml (ya configurado)

# 4. Servidor cámara (Windows PowerShell)
# $env:CAM_API_TOKEN = "3f7a1c2d-8e4b-4f9a-b1c2-d3e4f5a6b7c8"
# python camera_server.py   ← debe arrancar con host='0.0.0.0' para que WSL conecte

# Entrenamiento (en background)
cd /home/alvar/dagn_lib/training
/home/alvar/venv_tesis/bin/python -u train_fusion.py > /tmp/train_fusion.log 2>&1 &

# Evaluación
cd /home/alvar/dagn_lib
/home/alvar/venv_tesis/bin/python production/evaluate_fusion.py
```

---

## Estructura de archivos

```
dagn_lib/
├── CLAUDE.md                          ← este archivo
├── requirements.txt
├── results_log.txt                    ← historial de runs (best CCC por epoch)
├── production/
│   ├── fusion_model.py               ← FusionLSTM (81K params)
│   ├── fusion_best.pth               ← checkpoint activo (gitignored)
│   ├── analizar_emocion_service.py   ← FastAPI service (TODA la señal)
│   ├── dashboard.py                  ← Streamlit (solo visualización + MQTT)
│   └── evaluate_fusion.py            ← evaluación con CCC+IC95% por dataset
└── training/
    ├── train_fusion.py               ← script de entrenamiento principal
    ├── global_dataset.py             ← combina los 5 datasets, z-score por dataset
    ├── feature_extractor_face.py     ← MediaPipe AUs (17D)
    ├── feature_extractor_physio.py   ← NeuroKit2 HRV/EDA/TEMP (6D)
    ├── feature_extractor_eeg.py      ← MNE bandpower + asimetría (5D)
    ├── deap_dataset.py               ← DEAP loader
    ├── wesad_dataset.py              ← WESAD loader
    ├── dreamer_dataset.py            ← DREAMER loader
    ├── afew_va_dataset.py            ← AFEW-VA loader (usa .npy pre-extraídos)
    ├── affec_dataset.py              ← AFFEC loader (usa .npz pre-extraídos)
    ├── extract_afew_au_features.py   ← one-shot: PNG → AUs .npy (NO repetir)
    └── extract_affec_features.py     ← one-shot: AFFEC → .npz (NO repetir)
```

---

## Bugs conocidos y soluciones

| Problema | Solución |
|----------|----------|
| NeuroKit2 PPG: devuelve índices int64 no booleans | `_peaks_from_info()` detecta dtype antes de usar `np.where` |
| DREAMER .mat: `stimuli[stim_idx]` ES el array directamente | No acceder a sub-atributo `.EEG` |
| AFFEC TSV: algunos tienen cabecera | `pd.to_numeric(onset, errors='coerce') + dropna` |
| AFEW-VA JSON: está en `clip_dir/{clip_id}.json` | No buscar en el directorio padre |
| GlobalDataset: cargaba datasets dos veces | Cargar una vez con `_load_defaults` |
| PyTorch `strict=False`: falla con shape mismatch | Filtrar manualmente: `{k:v for k,v in state.items() if k in model_state and v.shape==model_state[k].shape}` |
| Dashboard 1.25 Hz → NeuroKit2 no detecta picos | Enviar `ir_batch[]` con todos los samples MQTT desde última llamada (50 Hz real) |
| `cam_hr` siempre 0 con cámara ~5 fps | Condición `nyq > 0.8` con `hr_hi = min(3.0, nyq*0.9)` en lugar de `nyq > 4.0` |
| Blinks nunca detectados | `ear_hires` usa AU43 del paso 2 (1 MediaPipe/llamada); Haar para rPPG en el bucle |
| SpO2 estancado en 83.7 al inicio | Usar solo últimos 5s del deque; reset deque al cambiar `session_id` |
| Valence desaparece del timeline | `key="va_timeline"`, `width=2`, `connectgaps=True`, `range=[-1.05,1.05]` |
| `st.image(url)` bloquea render loop | Usar `st.markdown('<img src=...>')` para MJPEG (browser gestiona el stream) |
| Streamlit muere al cerrar shell Claude | Lanzar con `setsid bash start_dashboard.sh` — crea nueva sesión de proceso |
| Token CAM_API_TOKEN no llega a procesos hijo | Usar `production/.streamlit/secrets.toml` leído natively por Streamlit |
| `record_start` falla con 409 al reiniciar sesión | Llamar `stop_camera` antes de `record_start` en START SESSION |
| `os.path.getmtime` FileNotFoundError en `/mnt/c/` | try/except OSError → fallback `time.time()` (race condition WSL/NTFS) |
| MediaPipe en bucle frames → 3s+ por llamada | Eliminar MediaPipe del bucle; solo Haar para rPPG; AU43 del paso 2 para blinks |

---

## Próximos pasos (2026-03-17)

### Estado de producción — SESIÓN EN DEPURACIÓN
Cambios aplicados hoy (2026-03-17):
- Dashboard usa `st.markdown(<img>)` en lugar de `st.image(url)` para cámara (evita bloqueo render)
- Proceso Streamlit lanzado con `setsid` + `start_dashboard.sh` (no muere al cerrar shell)
- Token cámara en `production/.streamlit/secrets.toml` (leído natively por Streamlit)
- `record_start` ahora llama `stop_camera` antes para evitar 409
- Error de `record_start` visible en sidebar (ya no falla silenciosamente)
- MediaPipe eliminado del bucle de frames (paso 4) — solo Haar para rPPG; AU43 reutilizado del paso 2
- `os.path.getmtime` con try/except OSError → fallback `time.time()` (race condition WSL/NTFS)
- Gráficas de señales: de 3 columnas → apiladas ancho completo
- Leyenda VA timeline: movida debajo de la gráfica (ya no se superpone)
- Logging INFO para face AU updates (nonzero/17, max value)

### ⚠️ BUG PENDIENTE — Warmup se queda al 3%
**Síntoma**: warmup muestra 3% (1/30) y no avanza.
**Diagnóstico**: la primera llamada al servicio completa OK, pero las siguientes timeout (>3s).
**Causa probable**: `_get_new_frames` con Haar en 30-60 frames acumulados en la primera llamada real
(con `_last_rppg_file=None` tras reset de sesión + cámara a ~30fps → 60 Haar × ~0.05s = 3s+).
**A investigar mañana**:
1. Añadir timing por paso en `_analyze_locked` para identificar cuál paso tarda
2. Posible fix: en el reset de sesión, inicializar `_last_rppg_file` al último frame existente
   (igual que `_advance_rppg_cursor` que se consideró antes) para que la primera llamada
   no procese frames acumulados sino solo los nuevos desde ese momento.
3. Alternativa: reducir cap de `_get_new_frames` de 60 a 5 frames por llamada

### Servidor cámara (Windows)
- Nueva versión con autenticación por token (`CAM_API_TOKEN`)
- DEBE arrancar con `host='0.0.0.0'` para que WSL conecte vía `172.26.96.1`
- Token: `3f7a1c2d-8e4b-4f9a-b1c2-d3e4f5a6b7c8` (también en `.streamlit/secrets.toml`)

---

## Notas importantes

- `fusion_best.pth` está en `.gitignore` — pesa mucho; compartir por otro medio si necesario
- WESAD local en `~/datasets/WESAD` (17 GB) — **borrar** con `rm -rf ~/datasets/WESAD` cuando no se entrene
- py-feat INCOMPATIBLE con Python 3.12 → usar MediaPipe FaceMesh (ya hecho)
- Dashboard llama al servicio a ~1.25 Hz (800ms autorefresh); warmup = 30 llamadas ≈ 24s
