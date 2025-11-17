import json
import re
import time
import torch
import unicodedata
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from app.domain.vocabulario import VOCABULARIO
from app.core.logging_config import logger
from app.core.config import settings
import torch.nn.functional as F


def quitar_tildes(s: str) -> str:
    s_protegido = s.replace('ñ', '\x00ENIE_MINUSCULA\x00').replace('Ñ', '\x00ENIE_MAYUSCULA\x00')
    
    nfd = unicodedata.normalize('NFD', s_protegido)
    sin_tildes = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    resultado = sin_tildes.replace('\x00ENIE_MINUSCULA\x00', 'ñ').replace('\x00ENIE_MAYUSCULA\x00', 'Ñ')
    
    return resultado


class BETOModismosAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        self.device = None

    def load_model(self):
        try:
            logger.info(f"Cargando modelo desde Hugging Face: {settings.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            self.setup_label_encoder()
            self.is_loaded = True
            logger.info("Modelo BETO-Finetuned cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando el modelo: {type(e).__name__}: {e}")
            self.is_loaded = False

    def setup_label_encoder(self):
        dataset_path = Path(settings.dataset_path)
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            data = [d for d in raw_data if isinstance(d, dict) and "etiquetas" in d]
            if not data:
                raise ValueError("El dataset no contiene etiquetas válidas.")

            all_labels = [et for entrada in data for et in entrada.get("etiquetas", [])]
            unique_labels = sorted(set(all_labels))

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(unique_labels)
            logger.info(f"LabelEncoder configurado con {len(unique_labels)} clases.")

        except FileNotFoundError:
            logger.warning("dataset.json no encontrado, usando clases por defecto.")
            fallback_classes = [
                'aburrido', 'agresión', 'animal', 'cambio', 'coito', 'comer',
                'comida', 'compañero', 'complicado', 'de mal gusto', 'de suerte',
                'dedicado', 'difícil', 'diligencia', 'diligencias', 'duro',
                'enojado', 'extraviar', 'falo', 'falso', 'gentilicio', 'gentilicio_femenino',
                'hartar', 'individuo torpe ', 'insoportable ', 'joven', 'joven_femenino', 'lenguaje',
                'lugar', 'malo', 'molestia', 'movimiento', 'movimientos', 'objeto', 'ordinario',
                'papeleta', 'perder', 'preminente', 'presumir', 'probar', 'riguroso',
                'salir', 'sartén', 'tarea', 'timar', 'tomar', 'trabajo', 'vagina'
            ]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(sorted(fallback_classes))

            logger.info("Clases por defecto configuradas:")
            for i, clase in enumerate(self.label_encoder.classes_, start=1):
                logger.info(f"{i}: {clase}")

        except Exception as e:
            logger.error(f"Error configurando LabelEncoder: {type(e).__name__}: {e}")

    def detectar_palabras(self, texto: str):
        texto_lower = texto.lower()
        encontradas = []

        # Extraer palabras con soporte para ñ y tildes
        palabras_texto = re.findall(r"\b[a-záéíóúüñ]+\b", texto_lower)

        for palabra_texto in palabras_texto:
            for palabra, raiz in VOCABULARIO.items():
                raiz_lower = raiz.lower()

                if palabra_texto.startswith(raiz_lower):
                    if "ñ" in raiz_lower and "ñ" not in palabra_texto:
                        continue
                    encontradas.append(palabra)

                else:
                    raiz_norm = quitar_tildes(raiz_lower)
                    palabra_norm = quitar_tildes(palabra_texto)
                    if palabra_norm.startswith(raiz_norm):
                        if "ñ" in raiz_lower and "ñ" not in palabra_texto:
                            continue
                        encontradas.append(palabra)

        if "vueltas" in encontradas and "vuelta" in encontradas:
            if re.search(r"\bvueltas\b", texto_lower):
                encontradas.remove("vuelta")
            else:
                encontradas.remove("vueltas")

        return encontradas

    def marcar_palabra_objetivo(self, texto: str, palabra_objetivo: str, todas_palabras: list):

        texto_normalizado = quitar_tildes(texto.lower())
        
        raices_a_enmascarar = []
        raiz_objetivo = None
        
        for palabra in todas_palabras:
            raiz = VOCABULARIO.get(palabra)
            if raiz:
                raiz_norm = quitar_tildes(raiz.lower())
                if palabra == palabra_objetivo:
                    raiz_objetivo = raiz_norm
                else:
                    raices_a_enmascarar.append(raiz_norm)
        
        texto_enmascarado = texto_normalizado
        for raiz in raices_a_enmascarar:
            patron = rf"\b{re.escape(raiz)}[a-záéíóúüñ]*\b"
            texto_enmascarado = re.sub(patron, "[MASK]", texto_enmascarado)
        
        if raiz_objetivo:
            patron_objetivo = rf"\b{re.escape(raiz_objetivo)}[a-záéíóúüñ]*\b"
            
            def reemplazo(match):
                return f"[TGT] {match.group(0)} [TGT]"
            
            resultado = re.sub(patron_objetivo, reemplazo, texto_enmascarado, count=1)
            
            if "[TGT]" not in resultado:
                logger.warning(f"No se marcó '{palabra_objetivo}'. Usando solo la palabra.")
                return f"[TGT] {raiz_objetivo} [TGT]"
            
            logger.debug(f"Texto preparado para '{palabra_objetivo}': {resultado}")
            return resultado
        
        # Fallback: solo la palabra objetivo
        return f"[TGT] {quitar_tildes(palabra_objetivo.lower())} [TGT]"

    def predecir_significado(self, texto_marcado: str):
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado.")

        texto_seguro = texto_marcado.encode("utf-8").decode("utf-8")

        inputs = self.tokenizer(
            texto_seguro,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        logger.debug(f"Tokens enviados al modelo: {tokens}")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            confianza, pred = torch.max(probs, dim=1)

        etiqueta = self.label_encoder.inverse_transform([pred.item()])[0]

        logger.info(f"Predicción -> Etiqueta: '{etiqueta}', Confianza: {confianza.item():.4f}, Texto: {texto_seguro}")

        return etiqueta, float(confianza.item())

    def analizar_texto(self, texto: str):
        start_time = time.time()
        palabras = self.detectar_palabras(texto)
        analizados = {}
        detalles = []

        logger.info(f"Texto recibido: '{texto}'")
        logger.info(f"Palabras detectadas para análisis: {palabras}")

        if not palabras:
            logger.info("No se detectaron modismos en el texto.")
            return {
                "texto_original": texto,
                "modismos_detectados": {},
                "modismos_detallados": [],
                "total_modismos": 0,
                "timestamp": start_time,
                "tiempo_procesamiento": 0.0,
                "modelo": "BETO-Finetuned"
            }

        for palabra in palabras:
            try:
                logger.debug(f"Analizando palabra: '{palabra}'")
                # Pasar todas las palabras detectadas para enmascarar correctamente
                marcado = self.marcar_palabra_objetivo(texto, palabra, palabras)
                significado, confianza = self.predecir_significado(marcado)

                analizados[palabra] = significado
                detalles.append({
                    "palabra": palabra,
                    "significado_detectado": significado,
                    "contexto": texto,
                    "confianza": str(round(confianza, 3))
                })

                logger.info(f"'{palabra}' → '{significado}' (confianza={confianza:.3f})")

            except Exception as e:
                logger.error(f"Error analizando '{palabra}': {type(e).__name__}: {e}")
                analizados[palabra] = f"error: {type(e).__name__}"

        tiempo_total = round(time.time() - start_time, 3)
        logger.info(f"Análisis completado en {tiempo_total}s. Modismos detectados: {len(analizados)}")

        return {
            "texto_original": texto,
            "modismos_detectados": analizados,
            "modismos_detallados": detalles,
            "total_modismos": len(analizados),
            "timestamp": start_time,
            "tiempo_procesamiento": tiempo_total,
            "modelo": "bert-cased-spanish-wsd-finetuned-modismos"
        }