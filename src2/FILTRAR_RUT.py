import pandas as pd
from tqdm import tqdm
import pickle
from fuzzywuzzy import fuzz, process
import numpy as np
import networkx as nx
from fuzzywuzzy import fuzz
from tqdm import tqdm
from collections import defaultdict
from unidecode import unidecode
import re
import unicodedata
from rapidfuzz import fuzz, process
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
import psycopg2
from psycopg2.extras import execute_batch
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from dateutil.relativedelta import relativedelta
import warnings
import psycopg2

# Desactiva warnings
warnings.filterwarnings("ignore")

url = r"base_grande.csv.xz"

conn_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'UnaCasaEnUnArbol2024',
    'host': '186.67.61.251'
}
# Crear la cadena de conexión
connection_string = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}/{conn_params['dbname']}"
# Crear el motor de SQLAlchemy
engine = create_engine(connection_string)
conn = psycopg2.connect(**conn_params)

tqdm.pandas()  # habilita barras en apply


def get_supermerge():
    df2 = pd.read_csv(url, compression="xz", sep="\t", usecols=['nombrecompleto_x', 'rut',"nombreencontrado"]).drop_duplicates()
    agrupado = df2.groupby("rut").size().reset_index()
    filtro = agrupado[agrupado[0] > 5].sort_values(by=0, ascending=False)
    filtro.to_excel("filtro_rut.xlsx", index=False)
    merge = filtro.merge(df2, on="rut")
    mask = merge["nombrecompleto_x"] == merge["nombreencontrado"]
    merge2 = merge[~mask]
    supermerge = pd.concat([merge[mask],merge[~mask]]).reset_index()
    del supermerge["index"]
    return supermerge, df2,merge,mask

def normalizar_nombre(nombre):
    if pd.isna(nombre):
        return ""
    nombre = str(nombre).upper()
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join(c for c in nombre if not unicodedata.combining(c))
    nombre = re.sub(r'[^A-Z ]', '', nombre)
    nombre = re.sub(r'\s+', ' ', nombre).strip()
    return nombre

def clave_agrupacion(nombre):
    tokens = sorted(normalizar_nombre(nombre).split())
    return ' '.join(tokens)



def aplicar_fuzzy_grupo(df_grupo, threshold=96, scorer_type='partial_ratio', 
                       show_progress=True, batch_size=100, prioridad_m=None):
    # Diccionario de scorers
    scorers = {
        'partial_ratio': fuzz.partial_ratio,
        'ratio': fuzz.ratio,
        'token_sort_ratio': fuzz.token_sort_ratio,
        'token_set_ratio': fuzz.token_set_ratio,
        'WRatio': fuzz.WRatio,
        'QRatio': fuzz.QRatio
    }
    
    if scorer_type not in scorers:
        raise ValueError(f"scorer_type debe ser uno de: {list(scorers.keys())}")
    
    scorer = scorers[scorer_type]
    #nombres =  df_grupo['nombrecompleto_x'].tolist()
    nombres =  df_grupo['nombrecompleto_x'].tolist()
    n = len(nombres)
    
    # 1. Precomputar índice de trigramas
    trigram_index = defaultdict(set)
    normalized_names = []
    
    for idx, nombre in enumerate(nombres):
        normalized = unidecode(nombre).lower().replace(" ", "")
        normalized_names.append(normalized)
        if len(normalized) >= 3:
            for i in range(len(normalized) - 2):
                trigram = normalized[i:i+3]
                trigram_index[trigram].add(idx)

    # 2. Construir grafo de similitud
    G = nx.Graph()
    G.add_nodes_from(range(n))
    processed_pairs = set()
    
    if show_progress:
        progress_bar = tqdm(total=n, desc="Construyendo grafo de similitud", unit="nombres")
    
    for i in range(n):
        # Obtener candidatos usando trigram_index
        candidates = set()
        normalized = normalized_names[i]
        
        if len(normalized) >= 3:
            for j in range(len(normalized) - 2):
                trigram = normalized[j:j+3]
                candidates.update(trigram_index.get(trigram, set()))
        else:
            candidates = set(range(n))
        
        # Eliminar pares ya procesados
        candidates = [j for j in candidates if j > i and (i, j) not in processed_pairs]
        
        # Comparar con candidatos
        for j in candidates:
            score = scorer(nombres[i], nombres[j])
            if score >= threshold:
                G.add_edge(i, j)
            processed_pairs.add((i, j))
        
        if show_progress:
            progress_bar.update(1)
            if (i + 1) % batch_size == 0:
                progress_bar.set_postfix({
                    'Aristas': G.number_of_edges(),
                    'Nodos': G.number_of_nodes()
                })
    
    if show_progress:
        progress_bar.close()
    
    # 3. Encontrar grupos usando componentes conectadas
    """
    groups = {}
    for comp in nx.connected_components(G):
        representative = min(comp, key=lambda idx: nombres[idx])
        for idx in comp:
            groups[idx] = representative
    """
    groups = {}
    for comp in nx.connected_components(G):
        if prioridad_m is not None:
            prioridad = [idx for idx in comp if idx < prioridad_m]
            if prioridad:
                representative = min(prioridad)
            else:
                representative = min(comp, key=lambda idx: nombres[idx])
        else:
            representative = min(comp, key=lambda idx: nombres[idx])
        
        for idx in comp:
            groups[idx] = representative
    
    # 4. Asignar grupos (incluye nodos no conectados)
    grupo_final = []
    for i in range(n):
        if i in groups:
            grupo_final.append(nombres[groups[i]])
        else:
            grupo_final.append(nombres[i])
    
    df_grupo = df_grupo.copy()
    df_grupo['grupo_final'] = grupo_final
    
    if show_progress:
        print(f"✅ Procesamiento completado:")
        print(f"   - Total nombres: {n}")
        print(f"   - Grupos creados: {len(set(grupo_final))}")
        print(f"   - Scorer usado: {scorer_type}")
        print(f"   - Threshold: {threshold}")
        print(f"   - Aristas en grafo: {G.number_of_edges()}")
    
    return df_grupo


def agrupar_nombres(df, col_nombres='nombres', salida='nombres_agrupados.csv',prioridad_m=None):
    print("🔍 Normalizando nombres...")
    df['nombre_normalizado'] = df[col_nombres].progress_apply(normalizar_nombre)

    print("🧠 Generando claves de agrupación...")
    df['clave'] = df['nombre_normalizado'].progress_apply(clave_agrupacion)
    df['nombrecompleto_x'] = df[col_nombres]  # para mantener compatibilidad

    print("🔗 Agrupando por similitud...")
    #df_resultado = df.groupby('clave', group_keys=False).progress_apply(aplicar_fuzzy_grupo)
    df_resultado = aplicar_fuzzy_grupo(df,prioridad_m=prioridad_m)

    print("✅ Agrupamiento completo.")
    return df_resultado[[col_nombres, 'grupo_final']]


def get_final():
    supermerge, df2,merge,mask = get_supermerge()
    salida = agrupar_nombres(supermerge,col_nombres="nombrecompleto_x",prioridad_m= len(merge[mask]))
    df3 = df2[["rut","nombreencontrado"]].drop_duplicates()
    salida2 = salida.merge(df3, left_on="grupo_final", right_on="nombreencontrado", how="left") 
    with open('contador.pkl', 'rb') as f:
        contador2 = pickle.load(f)
    
    contador2["procesados"]
    por_rutificar = salida2[salida2["rut"].isnull()][["nombrecompleto_x","grupo_final"]]
    solo_grupo = por_rutificar[["grupo_final"]].drop_duplicates()
    solo_grupo["rut"] = range(contador2["procesados"],contador2["procesados"]+len(solo_grupo))
    por_rutificar.merge(solo_grupo)
    rutificados = salida2[salida2["rut"].notnull()]
    final = pd.concat([rutificados,por_rutificar.merge(solo_grupo)])
    final["nombreencontrado"] = final["grupo_final"]
    del final["grupo_final"]
    return final
    
def get_df_filtrado(rut_filtro):
    # Ruts a filtrar
    rut_filtro_set = set(rut_filtro)
    
    
    # Inicializar lista para partes filtradas
    partes_filtradas = []
    
    # Primero contamos cuántas filas hay en total para estimar el número de chunks
    total_filas = sum(1 for _ in pd.read_csv(url, compression="xz", sep="\t", chunksize=1_000_000))
    chunk_size = 10000
    num_chunks_aprox = total_filas * 1000000 // chunk_size  # estimado conservador
    
    # Leer en chunks con barra de progreso
    for chunk in tqdm(pd.read_csv(url, compression="xz", sep="\t", chunksize=chunk_size), total=num_chunks_aprox, desc="Filtrando"):
        chunk_filtrado = chunk[chunk["rut"].isin(rut_filtro_set)]
        if not chunk_filtrado.empty:
            partes_filtradas.append(chunk_filtrado)
    
    # Concatenar los datos filtrados
    df_filtrado = pd.concat(partes_filtradas, ignore_index=True)
    return df_filtrado   

def df_final_corregido(final,df_filtrado):
    del df_filtrado["rut"]
    del df_filtrado["nombreencontrado"]
    merge = df_filtrado.merge(final)
    return merge

def limpiar_rut_base(rut_filtro_set):
    query = "DELETE FROM personal2 WHERE rut = ANY(%s::text[])"
    with conn.cursor() as cur:
        cur.execute(query, (list(rut_filtro_set),))
        conn.commit()

def insert_corregidos(merge):
    merge.to_sql("personal2",engine, index=False, if_exists="append")
    merge

def GLOBAL():
    print("Comenzando...")
    final = get_final()
    print("final cerrado")
    filtro_set = final["rut"].unique()
    df_filtrado = get_df_filtrado(filtro_set)
    print("df_filtrado cerrado")
    merge = df_final_corregido(final,df_filtrado)
    print("merge cerrado")
    limpiar_rut_base(filtro_set)
    print("limpieza cerrado")
    insert_corregidos(merge)
    print("insert cerrado")