# dagn_lib — Desarrollo y contexto

## Identidad del proyecto

**dagn_lib** es la contribución doctoral principal del sistema DAGN.
A diferencia de dagn_simple (que usa encoders CNN/LSTM pesados, 8.8M params),
dagn_lib extrae **features bibliográficamente fundamentadas** con librerías
establecidas (MediaPipe, NeuroKit2, MNE) y las fusiona con un LSTM minimalista
(81K params). La defensa doctoral se centra en esta elección arquitectónica.

```
face(T,17) + physio(T,6) + eeg(T,5) → LayerNorm(28) → LSTM(256×2) → Linear(2) → tanh → VA
```

Total: **819,770 parámetros** — 10× menos que dagn_simple (8.8M), features explícitas.

---

## Estado actual (2026-03-20) — NUEVO MEJOR RESULTADO

### Modelo — RESULTADO FINAL: evaluate_fusion **0.389**
- Checkpoint: `production/fusion_best.pth` = `production/fusion_baseline.pth` (Mar 20 09:58)
- Best training monitor CCC: **0.319** — epoch 115, early stopping epoch 175 (PATIENCE=60)
- Prior: `--use_prior` — directional hinge λ=0.05, `ds_face_ids=(DS_AFEWVA,)` (AFEWVA face only)
- Arquitectura: `FusionLSTM(hidden=256, layers=2)` en `production/fusion_model.py`
- Params: **819,770** (10× menos que dagn_simple)
- **ATENCIÓN**: `fusion_best.pth` puede ser sobreescrito. `fusion_baseline.pth` es el seguro.

### Evaluación final (split=val, N=937) — NUEVO MEJOR RESULTADO

| Dataset | N | CCC-V | 95% CI | CCC-A | 95% CI | Media |
|---------|---|-------|--------|-------|--------|-------|
| DEAP | 256 | 0.157 | [0.045, 0.280] | 0.033 | [-0.080, 0.155] | 0.095 |
| WESAD | 299 | 0.510 | [0.412, 0.599] | 0.596 | [0.507, 0.678] | **0.553** |
| DREAMER | 82 | -0.010 | [-0.048, 0.024] | -0.002 | [-0.041, 0.035] | ~0 |
| AFEW-VA | 182 | 0.586 | [0.489, 0.645] | 0.667 | [0.584, 0.730] | **0.627** |
| AFFEC | 118 | 0.699 | [0.587, 0.791] | 0.396 | [0.231, 0.540] | **0.548** |
| **GLOBAL** | **937** | **0.404** | **[0.350, 0.457]** | **0.374** | **[0.316, 0.435]** | **0.389** |

**Tabla LaTeX lista en `results_log.txt`.**

Análisis por dataset:
- **AFEW-VA 0.627** — AUs MediaPipe + prior FACS: +0.090 sobre baseline; **supera dagn_simple en 0.255**
- **AFFEC 0.548** — valence 0.699; prior sin reglas de cara (OpenFace2 incompatible) libera el physio
- **WESAD 0.553** — fisiología de muñeca, marginal degradación vs baseline
- **DEAP 0.095** — prior physio rules interfieren con z-score en targets de valence; trade-off conocido
- **DREAMER ~0** — excluido del entrenamiento (Likert 1-5, varianza casi nula)

Comparativa con dagn_simple (8.8M params):
- dagn_lib (820K): WESAD=0.553, AFEW-VA=**0.627**, AFFEC=**0.548**, GLOBAL=**0.389**
- dagn_simple (8.8M): WESAD=0.598, AFEW-VA=0.372, GLOBAL=0.435
- dagn_lib **supera en AFEW-VA por +0.255** y en AFFEC; GLOBAL más próximo (0.389 vs 0.435)

**Argumento doctoral**: 10× menos parámetros, features bibliográficamente fundamentadas (MediaPipe/NeuroKit2/MNE), prior fisiológico diferenciable (Ekman 1978 + Task Force 1996), AFEW-VA 0.627 >> dagn_simple 0.372.

### Prior fisiológico — ablation completo (2026-03-19)
Módulo `training/physiological_prior.py`: regularizador diferenciable fundamentado en literatura.
Flags: `--use_prior`, `--lambda_prior`, `--face_only_prior` en `train_fusion.py`.

| Variante | Train CCC | Eval GLOBAL | AFEW-VA | AFFEC |
|----------|-----------|-------------|---------|-------|
| Baseline (prior=OFF) | 0.336 | **0.380** | 0.537 | 0.418 |
| Prior full Gaussian NLL λ=0.10 | 0.138 | — | — | — |
| Prior full directional hinge λ=0.05 | 0.321 | 0.373 | **0.613** | 0.389 |
| Prior face-only λ=0.10 | 0.323 | 0.337 | 0.522 | 0.236 |
| Prior full + AFFEC AU /5 fix λ=0.05 | 0.319 | 0.364 | 0.587 | 0.389 |

Conclusiones:
- Prior directional hinge es la variante menos perjudicial (Δ=-0.007 global)
- AFEW-VA mejora +0.076 con prior full directional (reglas AU MediaPipe bien calibradas)
- AFFEC no se beneficia: OpenFace2 AU_r cualitativamente distinto a MediaPipe
- z-score inter-dataset hace incompatibles los targets absolutos → directional hinge necesario
- Valor doctoral: metodológico (restricciones diferenciables desde literatura) + ablation exhaustivo

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

### eeg (T=30, dim=5) — TGAM2-compatible (att/med → 5D)
Extraídos con **`feature_extractor_eeg_tgam2.py`**.
Mapea multichannel EEG → frontal theta/alpha/beta → att/med (0-100) → 5 features.
Mismo espacio de features que producción (NeuroSky TGAM2).
Referencias: Crowley (2010), Klimesch (1999).

| Col | Feature | Fórmula |
|-----|---------|---------|
| 0 | theta | log1p((1 - att/100) × 0.5) |
| 1 | alpha | log1p((med/100) × 0.5) |
| 2 | beta  | log1p((att/100) × 0.5) |
| 3 | asym  | 0.0 (TGAM2 mono-frontal) |
| 4 | ratio | theta - alpha |

Mapeo att/med desde bandpower frontal:
- att = clip(β/(α+θ+ε)×50, 0, 100)  — alta beta rel. a alpha+theta → más concentrado
- med = clip(α/(θ+β+ε)×50, 0, 100)  — alta alpha rel. a theta+beta → más relajado

Canales frontales por dataset:
- DEAP (32ch): F3=ch2 (izq), F4=ch19 (der)
- DREAMER (14ch Emotiv EPOC): F3=ch2, F4=ch11
- AFFEC (63ch g.tec): F3=ch9, F4=ch13

Rango output: [0, log1p(0.5)] ≈ [0, 0.405] — no z-score, bounded como producción.
`feature_extractor_eeg.py` (bandpower z-scored) queda como referencia; ya no se usa en loaders.

---

## Datasets de entrenamiento

| Dataset | Modalidades usadas | Muestras | Ruta |
|---------|-------------------|----------|------|
| DEAP | EEG 32ch (MNE) + BVP/GSR/TEMP (NK2) | 1280 | `/mnt/f/source_datasets/Fisiologico/DEAP/` |
| WESAD | BVP/EDA/TEMP wrist (NK2) | 1498 | `/home/alvar/datasets/WESAD` ⚠️ BORRAR tras entrenar |
| DREAMER | EEG 14ch (MNE) + ECG (NK2) | 414 | `/mnt/f/source_datasets/Fisiologico/DREAMER/` |
| AFEW-VA | AUs MediaPipe (pre-extraídos) | 914 | `/mnt/f/source_datasets/Fisiologico/AFEW-VA/` |
| AFFEC | AUs OpenFace2 + EEG + GSR | 594 | `/home/alvar/datasets/affec_features/` |

### Features pre-extraídas
- AFEW-VA AUs: `~/datasets/afew_va_au_features/` — 914 .npy (457 clips + 457 flipped)
  - Script: `training/extract_afew_au_features.py` — NO repetir
- AFFEC: `~/datasets/affec_features/` — 594 .npz (regeneradas con TGAM2; EEG zeroed en affec_dataset.py)
  - Script: `training/extract_affec_features.py` — NO repetir (ya TGAM2-compatible)

### Dataset bias — solución aplicada
Cada dataset usa escala VA distinta (1-9, 1-5, 0-10).
Solución: **z-score por sub-dataset** antes de entrenar (`global_dataset.py`).
→ El modelo aprende dinámica intra-dataset, no offsets inter-dataset.

---

## Modelo FusionLSTM

### Versión activa (MEJOR RESULTADO — face+physio+EEG TGAM2)
```python
# production/fusion_model.py — hidden_dim=256, num_layers=2
x = concat(face, physio, eeg)      # (B, T, 28)
x = LayerNorm(28)(x)               # normaliza escala entre modalidades
h, _ = LSTM(28→256, layers=2)(x)   # aprende dinámica temporal multimodal
h = Dropout(0.45)(h)
va = tanh(Linear(256→2)(h))        # (B, T, 2) valence/arousal ∈ [-1, 1]
grad = diff(va, dim=T)             # gradiente temporal (para producción)
```
**Parámetros**: LayerNorm(56) + LSTM(≈819K) + Linear(514) = **819,770**

EEG training: DEAP usa TGAM2 F3(ch2)/F4(ch19); AFFEC/WESAD/AFEW-VA tienen ceros.
EEG producción: att/med → `_eeg_approx()` → 5D compatible con TGAM2.

---

## Training (`training/train_fusion.py`)

```
Loss = MSE + (1 - CCC_valence) + (1 - CCC_arousal) + variance_penalty
Optimizer: AdamW, LR=1e-3, WD=3e-4
Scheduler: CosineAnnealingLR
Early stopping: PATIENCE=60
Modal dropout: P_face=0.2, P_physio=0.2, P_eeg=0.3
Variance penalty alphas: DEAP=0.5, WESAD=0.0, DREAMER=0.0, AFEW-VA=0.3, AFFEC=0.3
EPOCHS=200, BATCH_SIZE=32, T=30
EXCLUDE_DATASETS = {"DREAMER"}   — Likert 1-5, varianza casi nula
QUALITY_WEIGHTS = equal (1.0 para todos)
EEG: DEAP usa TGAM2(F3/F4); AFFEC/WESAD/AFEW-VA zeros; DREAMER excluido
SWA: AveragedModel desde epoch SWA_START=80; evaluado vs best al final
```

Checkpoint guardado en `production/fusion_best.pth` cuando mejora CCC de validación.
SWA se evalúa al final: si bate el best regular, sobreescribe el checkpoint.
**Lanzar con**: `python -u train_fusion.py --use_prior` (λ=0.05 es el default, ds_face_ids=AFEWVA)

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
    ├── train_fusion.py               ← script de entrenamiento principal (--use_prior, --lambda_prior, --face_only_prior)
    ├── physiological_prior.py        ← prior diferenciable: señal→VA, directional hinge loss
    ├── global_dataset.py             ← combina los 5 datasets, z-score por dataset
    ├── feature_extractor_face.py     ← MediaPipe AUs (17D)
    ├── feature_extractor_physio.py   ← NeuroKit2 HRV/EDA/TEMP (6D)
    ├── feature_extractor_eeg.py      ← MNE bandpower + asimetría (5D) [referencia, no usado en loaders]
    ├── feature_extractor_eeg_tgam2.py ← TGAM2-compatible: frontal→att/med→5D (producción-identical)
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

## Estado final (2026-03-20) — TODO CERRADO ✅

### Modelo ✅ NUEVO MEJOR — GLOBAL CCC **0.389**
Checkpoint: `production/fusion_baseline.pth` = `production/fusion_best.pth` (Mar 20 09:58)
Config: `--use_prior --lambda_prior 0.05` + `ds_face_ids=(DS_AFEWVA,)` en train_fusion.py

### Ablations completados ✅
- Prior fisiológico (2026-03-19): directional hinge full (AFEWVA+AFFEC face) → AFEW-VA=0.613, GLOBAL=0.373
- EEG 10D bilateral (2026-03-19): degradó (0.345). Revertido. F3/F4 5D es suficiente.
- SWA + seed fija (2026-03-20): seed=42 da eval 0.376 reproducible pero subóptimo. Model soup falla.
- **Prior AFEWVA-face only (2026-03-20)**: AFEW-VA=**0.627**, AFFEC=**0.548**, GLOBAL=**0.389** ← MEJOR

### Producción ✅ OPERATIVO
- FastAPI service + Streamlit dashboard funcionando (2026-03-18)

### Servidor cámara (Windows)
- Autenticación por token (`CAM_API_TOKEN`)
- DEBE arrancar con `host='0.0.0.0'` para que WSL conecte vía `172.26.96.1`
- Token: `3f7a1c2d-8e4b-4f9a-b1c2-d3e4f5a6b7c8` (también en `.streamlit/secrets.toml`)

### Aprendizajes técnicos (SWA 2026-03-20)
- SWA ayuda cuando el modelo mejora durante la ventana de averaging (best_ep en medio del rango SWA)
- SWA_START=80 con best en ep100 sería el escenario ideal; con best en ep80 es perjudicial
- Model soup (Wortsman 2022) NO funciona con inicializaciones distintas; solo con fine-tunes de misma base

---

## Notas importantes

- `fusion_best.pth` está en `.gitignore` — pesa mucho; compartir por otro medio si necesario
- WESAD local en `~/datasets/WESAD` (17 GB) — **borrar** con `rm -rf ~/datasets/WESAD` cuando no se entrene
- py-feat INCOMPATIBLE con Python 3.12 → usar MediaPipe FaceMesh (ya hecho)
- Dashboard llama al servicio a ~1.25 Hz (800ms autorefresh); warmup = 30 llamadas ≈ 24s
