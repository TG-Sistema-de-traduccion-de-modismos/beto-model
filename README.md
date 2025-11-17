# **BETO Model Service**

**Servicio de inferencia para detección y clasificación de modismos en texto.**  
El modelo es una variante fine‑tuned sobre la base **dccuchile/bert-base-spanish-wwm-cased**; por simplicidad en la documentación se le denominará **"BETO"**.

## **Resumen**
**BETO** expone endpoints HTTP mínimos para:
- **cargar** el modelo fine‑tuned,
- **comprobar el estado** (health),
- **obtener predicciones** sobre textos que contengan modismos.

Este repositorio contiene únicamente la lógica para **cargar el checkpoint fine‑tuned (basado en dccuchile/bert-base-spanish-wwm-cased)**, realizar inferencia y exponer los endpoints HTTP (FastAPI). El fine‑tuning se realizó para **clasificar exactamente 21 modismos**; la API devuelve únicamente esas clases.

---

## **Modelo y alcance**
- **Base del modelo:** dccuchile/bert-base-spanish-wwm-cased (BERT cased en español).  
- **Fine‑tuning:** orientado a identificar el **significado contextual** de modismos en oraciones.  
- **Clases:** **21 modismos** (cada uno con sus significados posibles).  
- **Alcance del código:** solo carga el modelo, preprocesa texto, detecta modismos, realiza inferencia y expone `/health` y `/predict`. No incluye entrenamiento.

**Nota:** la imagen Docker resultante pesa aproximadamente **19.4 GB** después del build.

---

## **Objetivo del fine‑tuning**
La idea es que BETO sea capaz de identificar **qué significado** tiene un modismo según el contexto de la oración (por ejemplo, distinguir "camello" = trabajo vs "camello" = animal).

---

## **Modismos y significados (21)**

| # | **Modismo** | **Significados** |
|---:|-------------|------------------|
| 1 | **Arepa** | De suerte; Vagina; Comida; Individuo Torpe |
| 2 | **Berraco** | Dedicado; Enojado; Difícil |
| 3 | **Boleta** | De mal gusto; Papeleta |
| 4 | **Cacharrear** | Probar |
| 5 | **Camello** | Trabajo; Animal |
| 6 | **Chicanear** | Presumir |
| 7 | **Chimbo** | Malo; Falso; Falo |
| 8 | **Chino** | Joven; Gentilicio; Lenguaje |
| 9 | **China** | Joven (fem.); Gentilicio (fem.); Lugar |
|10 | **Desparchado** | Aburrido |
|11 | **Embolatar** | Extraviar; Timar; Perder (perder el tiempo) |
|12 | **Jartar** | Tomar (alcohol); Comer; Hartar |
|13 | **Joda** | Perturbación; Objeto |
|14 | **Mamera** | Pereza |
|15 | **Mamon** | Insoportable |
|16 | **Ñero** | Ordinario; Compañero |
|17 | **Paila** | Resignación; Sartén |
|18 | **Parchar** | Salir |
|19 | **Severo** | Preminente; Riguroso; Duro |
|20 | **Vuelta** | Diligencia; Coito; Agresión; Tarea; Movimiento |
|21 | **Vueltas** | Diligencias; Cambio; Movimientos |

---

## **Endpoints**
- **GET /health**  
  Retorna JSON con: `status`, `model_loaded`, `gpu_available`, `gpu_name`.
- **POST /predict**  
  Payload: JSON con el texto a analizar.  
  Respuesta: JSON con modismos detectados (hasta las 21 clases) y metadatos básicos.

Ejemplo:
```sh
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"texto":"Tu frase aquí"}'
```

---

## **Docker — build & run usando Dockerfile y GPU**
Requisitos en host: Docker 19.03+ y **NVIDIA Container Toolkit** (o runtime nvidia legacy). Comandos desde PowerShell o CMD en Windows:

1) Construir la imagen:
```sh
docker build -t beto-model:latest ./beto-model
```
> **Nota:** la imagen resultante pesa aproximadamente **19.4 GB**.


> **Nota:** esta imagen de Docker fue especialmente hecha para su uso con una RTX 5070.


2) Ejecutar (opción moderna --gpus, recomendada):
```sh
docker run --rm --name beto-model `
  --gpus all `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -p 8002:8002 `
  -v C:\ruta\al\modelo:/app/model `
  beto-model:latest
```
---

## **Configuración importante**
- **Revisa `requirements.txt`** y usa las versiones indicadas (torch, transformers, tokenizers, etc.). Las incompatibilidades entre versiones suelen ser la causa principal de errores en carga o inferencia.  
- Si usas archivos de configuración o `.env`, **actualiza las IPs/hosts/puertos** en `app/core/config.py` y en `.env` si tu despliegue no usa `localhost` (p. ej. `SERVICE_HOST`, `MODEL_HOST`).  
---

## **Limitaciones y recomendaciones**
- El repositorio solo **carga** y **sirve** el modelo fine‑tuned; **no incluye** el pipeline de entrenamiento.  
- Ejecutar en GPU con suficiente VRAM para evitar OOM al cargar modelos grandes.  
- Montar el checkpoint externamente si quieres reducir el tamaño de la imagen o facilitar actualizaciones sin rebuild.  
- Revisar logs (logger) para diagnosticar errores de carga o inferencia.
