import json
from factura_tool import extraer_datos_factura

def simular_llamada_ia(nombre_prueba: str, argumentos: dict):
    print(f"\n=== PRUEBA: {nombre_prueba} ===")
    
    # Así es exactamente como el orquestador ejecutará tu función
    resultado = extraer_datos_factura(**argumentos)
    
    # Verificamos que el resultado sea 100% serializable a JSON
    # (si aquí falla, al orquestador también le fallaría)
    try:
        json_salida = json.dumps(resultado, ensure_ascii=False, indent=2)
        print(json_salida)
    except TypeError as e:
        print(f"[FALLO DE SERIALIZACIÓN] La tool devolvió algo que no es JSON: {e}")


if __name__ == "__main__":
    # 1. Caso exitoso: archivo real (usando tu PDF de ejemplo)
    simular_llamada_ia(
        "Factura válida (solo devolver datos)",
        {
            "input_path": "ITE430714KI0_XAXX010101000_ACEM_479011_E9F8F496-1DB8-4361-975B-9B9D2C51FCEE.pdf",
            "page_num": 1
        }
    )

    # 2. Caso exitoso guardando archivo en disco
    simular_llamada_ia(
        "Factura válida (guardando JSON)",
        {
            "input_path": "ITE430714KI0_XAXX010101000_ACEM_479011_E9F8F496-1DB8-4361-975B-9B9D2C51FCEE.pdf",
            "page_num": 1,
            "output_json_path": "factura_extraida.json"
        }
    )

    # 3. Caso de error: archivo inexistente (NO debe detener el programa)
    simular_llamada_ia(
        "Archivo que no existe",
        {
            "input_path": "ruta/falsa/no_existe.pdf"
        }
    )

    # 4. Caso de error: página inválida (NO debe detener el programa)
    simular_llamada_ia(
        "Número de página fuera de rango",
        {
            "input_path": "ITE430714KI0_XAXX010101000_ACEM_479011_E9F8F496-1DB8-4361-975B-9B9D2C51FCEE.pdf",
            "page_num": 99
        }
    )