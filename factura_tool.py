import os
import re
import json
import tempfile
from typing import Optional, Any
import pymupdf
import ollama
    
# 1. PLANTILLA JSON DE REFERENCIA

ESTRUCTURA_JSON_EJEMPLO = {
    "fecha_y_hora_de_emision": "YYYY-MM-DDTHH:MM:SS",
    "folio_fiscal": "UUID de la factura",
    "emisor": {
        "rfc": "",
        "regimen_fiscal": "",
        "lugar_de_expedicion": ""
    },
    "receptor": {
        "uso_cfdi": "",
        "rfc": "",
        "nombre": "",
        "regimen_fiscal": "",
        "domicilio_fiscal": ""
    },
    "conceptos": [
        {
            "cantidad": 0.0,
            "clave_producto_o_servicio": "",
            "unidad": "",
            "descripcion": "",
            "objeto_impuesto": "",
            "impuesto": "",
            "valor_unitario": 0.0,
            "importe": 0.0,
            "descuento": 0.0
        }
    ],
    "subtotal": 0.0,
    "moneda": "MXN",
    "total": 0.0,
    "metodo_de_pago": ""
}


# ==========================================
# 2. FUNCIONES INTERNAS DEL PIPELINE
# ==========================================

def _prepare_input_image(input_path: str, page_number: int = 1, dpi: int = 200) -> tuple[str, bool]:
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(
            f"No se encontró el archivo en la ruta '{input_path}'. Verifica que la ruta sea correcta."
        )

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        with pymupdf.open(input_path) as doc:
            total_pages = len(doc)
            if page_number < 1 or page_number > total_pages:
                raise ValueError(
                    f"El número de página {page_number} no es válido. El PDF tiene {total_pages} página(s)."
                )
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(dpi=dpi)

            # Archivo temporal único para evitar colisiones en llamadas concurrentes
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_png = tmp.name
            tmp.close()
            pix.save(temp_png)

        return temp_png, True

    elif ext in [".jpg", ".jpeg", ".png"]:
        return input_path, False

    else:
        raise ValueError(
            f"Formato '{ext}' no soportado. Proporciona un archivo .pdf, .png, .jpg o .jpeg."
        )


def _ocr_image_to_text(image_path: str, ocr_model: str = "glm-ocr:latest") -> str:
    response = ollama.chat(
        model=ocr_model,
        messages=[
            {
                "role": "user",
                "content": "Text Recognition:",
                "images": [image_path]
            }
        ],
        options={
            "temperature": 0.0,
            "num_ctx": 16384,
            "num_predict": 4096
        }
    )

    ocr_text = response["message"]["content"].strip()
    if not ocr_text:
        raise RuntimeError(
            "El motor OCR devolvió un texto vacío. La imagen o página del PDF podría estar en blanco o ilegible."
        )
    return ocr_text


def _extract_json_safely(raw_content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def _parse_text_to_json(raw_ocr_text: str, llm_model: str = "qwen3.5:2b") -> dict[str, Any]:
    esquema_str = json.dumps(ESTRUCTURA_JSON_EJEMPLO, ensure_ascii=False, indent=2)

    prompt = (
        "Eres un extractor de datos fiscales mexicanos (CFDI). "
        "Analiza el texto de la factura y devuelve ÚNICAMENTE un objeto JSON válido "
        "que siga exactamente esta estructura y nombres de llaves:\n\n"
        f"{esquema_str}\n\n"
        "Reglas:\n"
        "- Convierte los importes numéricos a float sin símbolo '$' ni comas (ejemplo: 1450.50).\n"
        "- Si un campo de texto no aparece en la factura, pon \"\".\n"
        "- Si un campo numérico no aparece (como descuento), pon 0.0.\n"
        "- Incluye todos los productos o servicios dentro de la lista 'conceptos'.\n\n"
        f"TEXTO DE LA FACTURA:\n{raw_ocr_text}"
    )

    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        think=False,
        options={
            "temperature": 0.0,
            "num_ctx": 16384,
            "num_predict": 4096
        }
    )

    raw_output = response["message"]["content"].strip()

    if not raw_output:
        response = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={
                "temperature": 0.0,
                "num_ctx": 16384,
                "num_predict": 4096
            }
        )
        raw_output = response["message"]["content"].strip()

    return _extract_json_safely(raw_output)


# ==========================================
# 3. ENTRYPOINT DE LA TOOL (PARA EL ORQUESTADOR)
# ==========================================

def extraer_datos_factura(
    input_path: str,
    page_num: int = 1,
    output_json_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Tool ejecutable por el orquestador.
    Siempre retorna un diccionario serializable a JSON con 'status': 'success' o 'error'.
    """
    # Validación defensiva de parámetros generados por el modelo
    if not isinstance(input_path, str) or not input_path.strip():
        return {
            "status": "error",
            "error_type": "invalid_argument",
            "message": "El parámetro 'input_path' debe ser una cadena de texto con la ruta del archivo."
        }

    if not isinstance(page_num, int) or page_num < 1:
        return {
            "status": "error",
            "error_type": "invalid_argument",
            "message": "El parámetro 'page_num' debe ser un número entero mayor o igual a 1."
        }

    image_path: Optional[str] = None
    is_temp = False

    try:
        image_path, is_temp = _prepare_input_image(input_path, page_number=page_num, dpi=200)
        ocr_text = _ocr_image_to_text(image_path, ocr_model="glm-ocr:latest")
        datos_factura = _parse_text_to_json(ocr_text, llm_model="qwen3.5:2b")

        resultado: dict[str, Any] = {
            "status": "success",
            "data": datos_factura
        }

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(datos_factura, f, ensure_ascii=False, indent=4)
            resultado["saved_to"] = output_json_path

        return resultado

    except FileNotFoundError as e:
        return {
            "status": "error",
            "error_type": "file_not_found",
            "message": str(e)
        }
    except ValueError as e:
        return {
            "status": "error",
            "error_type": "invalid_input",
            "message": str(e)
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_type": "parsing_failed",
            "message": "El modelo interno no pudo estructurar el texto extraído en un JSON válido. Intenta de nuevo o verifica la calidad del documento."
        }
    except ollama.ResponseError as e:
        return {
            "status": "error",
            "error_type": "ollama_error",
            "message": f"Error en el servidor de Ollama ({e.status_code}): {e.error}. Verifica que los modelos 'glm-ocr:latest' y 'qwen3.5:2b' estén descargados."
        }
    except ConnectionError:
        return {
            "status": "error",
            "error_type": "service_unavailable",
            "message": "No se pudo conectar con el servicio local de Ollama. Asegúrate de que esté en ejecución."
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": f"Ocurrió un error inesperado al procesar la factura: {str(e)}"
        }
    finally:
        if is_temp and image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass