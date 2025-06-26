import pandas as pd
import os


url = r"../fix_historial3\salida\salida"


def listar_archivos(carpeta):
    try:
        # Obtener la lista de todos los archivos y directorios en la carpeta
        archivos = os.listdir(carpeta)
        
        # Filtrar solo los archivos (no subdirectorios)
        archivos = [f for f in archivos if os.path.isfile(os.path.join(carpeta, f))]
        
        return archivos
    except FileNotFoundError:
        print(f"La carpeta '{carpeta}' no existe.")
        return []
    except Exception as e:
        print(f"Error al listar archivos: {e}")
        return []
    
    
def unir_base_grande():
    
    files = listar_archivos(url)
    output_file = "base_grande.csv.xz"

    # Elimina el archivo de salida si ya existe (opcional)
    if os.path.exists(output_file):
        os.remove(output_file)

    for i, file in enumerate(files):
        path = f"{url}/{file}"
        df = pd.read_csv(path, compression='xz', sep='\t')
        
        # Solo escribe el header en la primera iteración
        df.to_csv(output_file, mode='a', compression='xz', sep='\t', index=False, header=(i == 0))
        
        # Liberar memoria explícitamente
        del df
