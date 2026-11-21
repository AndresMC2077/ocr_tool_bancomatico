import os
import re
import json
import pymupdf
import ollama

# ==========================================
# 1. PLANTILLA JSON DE REFERENCIA
# ==========================================

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
# 2. FUNCIONES DEL PIPELINE
# ==========================================

def prepare_input_image(input_path: str, page_number: int = 1, dpi: int = 200) -> tuple[str, bool]:
    """
    Convierte PDF a PNG temporal o usa la imagen JPG/PNG directamente.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        temp_png = f"temp_page_{page_number}.png"
        with pymupdf.open(input_path) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"El número de página debe estar entre 1 y {len(doc)}")
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(dpi=dpi)
            pix.save(temp_png)
        print(f"[1/3] Página {page_number} del PDF convertida a '{temp_png}' ({dpi} DPI).")
        return temp_png, True

    elif ext in [".jpg", ".jpeg", ".png"]:
        print(f"[1/3] Imagen detectada directamente: '{input_path}'.")
        return input_path, False

    else:
        raise ValueError(f"Formato no soportado ({ext}). Usa .pdf, .png, .jpg o .jpeg.")


def ocr_image_to_text(image_path: str, ocr_model: str = "glm-ocr:latest") -> str:
    """
    Extrae todo el texto de la factura usando glm-ocr.
    """
    print(f"[2/3] Extrayendo texto de la factura con '{ocr_model}'...")
    
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
        raise RuntimeError("glm-ocr devolvió un texto vacío. Verifica que la imagen sea legible.")
        
    print(f"      -> OCR completado ({len(ocr_text)} caracteres extraídos).")
    return ocr_text


def extract_json_safely(raw_content: str) -> dict:
    """
    Limpia etiquetas <think> o bloques ```json si el modelo los incluye, y parsea el JSON.
    """
    # Eliminar posibles bloques <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    
    # Eliminar bloques markdown ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Si aún hay texto basura alrededor, buscar desde la primera '{' hasta la última '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def parse_text_to_json(raw_ocr_text: str, llm_model: str = "qwen3.5:2b") -> dict:
    """
    Estructura el texto extraído en JSON usando Qwen 3.5 con think=False.
    """
    print(f"[3/3] Estructurando datos en JSON con '{llm_model}'...")
    
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
    
    # Llamada desactivando el modo thinking para evitar que devuelva content vacío
    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        think=False,  # Evita que Qwen 3.5 agote los tokens pensando o choque con format="json"
        options={
            "temperature": 0.0,
            "num_ctx": 16384,
            "num_predict": 4096
        }
    )
    
    raw_output = response["message"]["content"].strip()
    
    # Respaldo por si la versión de ollama o el modelo ignora format="json"
    if not raw_output:
        print("      [!] Respuesta vacía con format='json', reintentando en modo directo...")
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

    try:
        return extract_json_safely(raw_output)
    except json.JSONDecodeError as e:
        print("\n[ERROR] No se pudo decodificar la respuesta de Qwen. Respuesta cruda recibida:")
        print(raw_output)
        raise e


def run_invoice_pipeline(input_path: str, output_json_path: str, page_num: int = 1):
    image_path, is_temp = prepare_input_image(input_path, page_number=page_num, dpi=200)
    
    try:
        # Paso 2: Imagen a Texto con glm-ocr
        ocr_text = ocr_image_to_text(image_path, ocr_model="glm-ocr:latest")
        
        # Paso 3: Texto a JSON con qwen3.5:2b
        datos_factura = parse_text_to_json(ocr_text, llm_model="qwen3.5:2b")
        
        # Guardar archivo JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(datos_factura, f, ensure_ascii=False, indent=4)
            
        print(f"Proceso completado. Factura guardada en '{output_json_path}'.")
        return datos_factura

    finally:
        if is_temp and os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    ARCHIVO_ENTRADA = "ITE430714KI0_XAXX010101000_ACEM_479011_E9F8F496-1DB8-4361-975B-9B9D2C51FCEE.pdf"
    ARCHIVO_SALIDA = "factura_extraida.json"
    
    run_invoice_pipeline(
        input_path=ARCHIVO_ENTRADA,
        output_json_path=ARCHIVO_SALIDA,
        page_num=1
    )