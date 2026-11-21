import json
import ollama
from factura_tool import extraer_datos_factura

# 1. El contrato en el formato que espera Ollama
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extraer_datos_factura",
            "description": (
                "Extrae y estructura los datos fiscales de una factura mexicana (CFDI) "
                "a partir de un archivo PDF o imagen (PNG, JPG, JPEG). Devuelve emisor, "
                "receptor, conceptos, importes y folio fiscal en JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Ruta del archivo de la factura (.pdf, .png, .jpg, .jpeg)."
                    },
                    "page_num": {
                        "type": "integer",
                        "description": "Número de página si es PDF (empieza en 1)."
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Ruta opcional para guardar el JSON en disco."
                    }
                },
                "required": ["input_path"]
            }
        }
    }
]

MODELO_AGENTE = "qwen3.5:2b"
PDF_PRUEBA = "ITE430714KI0_XAXX010101000_ACEM_479011_E9F8F496-1DB8-4361-975B-9B9D2C51FCEE.pdf"


def ejecutar_prueba(pregunta_usuario: str):
    messages = [{"role": "user", "content": pregunta_usuario}]
    print(f"Usuario: {pregunta_usuario}\n")

    # Paso 1: Enviar pregunta + contrato al modelo
    response = ollama.chat(
        model=MODELO_AGENTE,
        messages=messages,
        tools=TOOLS,
        think=False
    )

    mensaje_ia = response["message"]
    tool_calls = mensaje_ia.get("tool_calls")

    # Si el modelo no quiso usar ninguna tool, imprimimos su respuesta directa
    if not tool_calls:
        print("IA (sin usar tools):", mensaje_ia["content"])
        return

    # Guardamos la petición de tool en el historial
    messages.append(mensaje_ia)

    # Paso 2: Ejecutar las tools que pidió el modelo
    for call in tool_calls:
        nombre = call["function"]["name"]
        argumentos = call["function"]["arguments"]
        print(f"[Orquestador] La IA pidió ejecutar '{nombre}' con: {argumentos}")

        if nombre == "extraer_datos_factura":
            resultado = extraer_datos_factura(**argumentos)
        else:
            resultado = {"status": "error", "message": f"Tool desconocida: {nombre}"}

        print(f"[Orquestador] Resultado devuelto a la IA: status={resultado.get('status')}\n")

        # Paso 3: Devolver el resultado JSON a la IA
        messages.append({
            "role": "tool",
            "content": json.dumps(resultado, ensure_ascii=False)
        })

    # Paso 4: Pedirle a la IA la respuesta final ya con los datos en contexto
    respuesta_final = ollama.chat(
        model=MODELO_AGENTE,
        messages=messages,
        think=False
    )
    print("IA:", respuesta_final["message"]["content"])


if __name__ == "__main__":
    # Prueba 1: Debería disparar la tool con tu PDF de ejemplo
    ejecutar_prueba(
        f"Revisa la factura '{PDF_PRUEBA}' y dime el RFC del emisor, el folio fiscal y el total."
    )

    print("\n" + "-" * 50 + "\n")

    # Prueba 2: Debería intentar usar la tool, recibir tu error controlado y avisarte
    ejecutar_prueba(
        "Extrae el total de la factura 'no_existe.pdf'."
    )