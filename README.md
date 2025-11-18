# **BETO Model Service**

**Servicio de inferencia para detección y clasificación de modismos en texto.**

---

## **Resumen**

**BETO Model Service** expone endpoints HTTP mínimos para:
- **Cargar** el modelo fine‑tuned
- **Comprobar el estado** (health)
- **Obtener predicciones** sobre textos que contengan modismos

Este repositorio contiene únicamente la lógica para **cargar el checkpoint fine‑tuned**, realizar inferencia y exponer los endpoints HTTP (FastAPI). El fine‑tuning se realizó para **clasificar exactamente 21 modismos**; la API devuelve únicamente esas clases.

---

## **Modelo y arquitectura**

### **Modelo Base**
- **Nombre:** BETO (BERT en español)
- **Repositorio:** [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
- **Descripción:** Modelo BERT entrenado en un corpus grande en español utilizando la técnica de Whole Word Masking (WWM). Es de tamaño similar a BERT-Base y supera a Multilingual BERT en varios benchmarks del idioma español.

### **Modelo Fine-tuned**
- **Nombre:** BETO Fine-tuned para Modismos Bogotanos
- **Repositorio:** [pescobarg/BETO-finetuned-modismos](https://huggingface.co/pescobarg/BETO-finetuned-modismos)
- **Descripción:** Versión ajustada específicamente para identificar y clasificar 21 modismos del español bogotano según su significado contextual.
- **Tarea:** Clasificación multiclase de modismos en contexto

### **Alcance del código**
Este repositorio solo:
- Carga el modelo fine‑tuned
- Preprocesa texto
- Detecta modismos
- Realiza inferencia
- Expone endpoints `/health` y `/predict`

**No incluye el proceso de entrenamiento/fine-tuning.**

---

## **Objetivo del fine‑tuning**

El modelo es capaz de identificar **qué significado** tiene un modismo según el contexto de la oración. Por ejemplo, distinguir:
- "camello" = trabajo vs "camello" = animal
- "arepa" = suerte vs "arepa" = comida

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

### **GET /health**
Retorna JSON con: `status`, `model_loaded`, `gpu_available`, `gpu_name`.

**Ejemplo:**
```sh
curl -X GET http://localhost:8002/health
```

### **POST /predict**
Payload: JSON con el texto a analizar.  
Respuesta: JSON con modismos detectados (hasta las 21 clases) y metadatos básicos.

**Ejemplo:**
```sh
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"texto":"Tengo mucho camello esta semana"}'
```

---

## **Docker — build & run usando Dockerfile y GPU**

### **Requisitos**
- Docker 19.03+
- **NVIDIA Container Toolkit** (o runtime nvidia legacy)
- GPU compatible (optimizado para RTX 5070)

### **1. Construir la imagen**
```sh
docker build -t beto-model:latest ./beto-model
```

> **Nota:** La imagen resultante pesa aproximadamente **19.4 GB**.

> **Nota:** Esta imagen de Docker fue especialmente diseñada para su uso con una RTX 5070.

### **2. Ejecutar el contenedor**
Opción moderna `--gpus` (recomendada):
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
- Si usas archivos de configuración o `.env`, **actualiza las IPs/hosts/puertos** en `app/core/config.py` y en `.env` si tu despliegue no usa `localhost` (por ejemplo: `SERVICE_HOST`, `MODEL_HOST`).

---

## **Limitaciones y recomendaciones**

- El repositorio solo **carga** y **sirve** el modelo fine‑tuned; **no incluye** el pipeline de entrenamiento.
- Ejecutar en GPU con suficiente VRAM para evitar errores OOM al cargar modelos grandes.
- Montar el checkpoint externamente si quieres reducir el tamaño de la imagen o facilitar actualizaciones sin rebuild.
- Revisar logs (logger) para diagnosticar errores de carga o inferencia.

