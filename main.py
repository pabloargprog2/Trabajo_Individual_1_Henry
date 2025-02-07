# uvicorn main:app --reload
# http://localhost:8000/docs/

import pandas as pd
from fastapi import FastAPI,HTTPException 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a mi API con FastAPI!"}

# Función para obtener la cantidad de filmaciones por mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    dataset = pd.read_csv("./dataset_funciones_basicas.csv")
    
    # Convertir la columna a datetime
    dataset['release_date'] = pd.to_datetime(dataset['release_date'], errors='coerce')

    # Diccionario para convertir los nombres de los meses en español a números
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    
    # Validar el mes ingresado
    mes = mes.lower()
    if mes not in meses:
        return {"error": "Mes inválido. Por favor ingresa un mes en español."}
    
    # Obtener el número del mes
    mes_num = meses[mes]
    
    # Filtrar las películas por mes
    peliculas_mes = dataset[dataset["release_date"].dt.month == mes_num]
    cantidad = len(peliculas_mes)
    
    # Retornar la cantidad de películas
    return {
        "message": f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}."
    }
    
     
     
# Diccionario para convertir los días de la semana en español a números (lunes = 0, domingo = 6)
dias = {
    "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6
}
# Función para obtener la cantidad de filmaciones por día
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    
    dataset = pd.read_csv("./dataset_funciones_basicas.csv")
    
    # Convertir la columna a datetime
    dataset['release_date'] = pd.to_datetime(dataset['release_date'], errors='coerce')
    
    # Validar el día ingresado
    dia = dia.lower()
    if dia not in dias:
        return {"error": "Día inválido. Por favor ingresa un día en español."}
    
    # Obtener el número del día
    dia_num = dias[dia]
    
    # Filtrar las películas por día de la semana
    peliculas_dia = dataset[dataset["release_date"].dt.weekday == dia_num]
    cantidad = len(peliculas_dia)
    
    # Retornar la cantidad de películas
    return {
        "message": f"{cantidad} cantidad de películas fueron estrenadas en el día {dia.capitalize()}."
    }
    
    
    # Funcion que recibe el título de una filmación
@app.get("/score_titulo/")
def score_titulo(titulo_de_la_filmacion: str):
    
    dataset = pd.read_csv("./dataset_funciones_basicas.csv")
    
    # Filtrar el DataFrame para encontrar la película con el título dado (ignorando mayúsculas y minúsculas)
    pelicula = dataset[dataset['title'].str.lower() == titulo_de_la_filmacion.lower()]

    # Si no se encuentra la película, lanzar una excepción HTTP 404
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Extraer los datos necesarios de la primera coincidencia
    titulo = pelicula.iloc[0]['title']  # Título de la película
    anio = int(pelicula.iloc[0]['release_year'])  # Año de estreno
    score = float(pelicula.iloc[0]['vote_average'])  # Puntaje promedio de la película

    # Devolver la información
    return {
        "titulo": titulo,
        "anio_estreno": anio,
        "score": score
    }
    
  # Funcion que recibe un titulo y devuelve el título, la cantidad de votos y el promedio de votaciones de una filmación.  
@app.get("/votes/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    dataset = pd.read_csv("./dataset_funciones_basicas.csv")
    
    # Filtrar el dataset por título
    film = dataset[dataset["title"].str.lower() == titulo_de_la_filmacion.lower()]

    # Validar si la película existe
    if film.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")

    # Extraer la cantidad de votos y el promedio
    votes = film.iloc[0]["vote_count"]
    average = film.iloc[0]["vote_average"]
    title = film.iloc[0]["title"]

    # Verificar si cumple con el umbral de 2000 votos
    if votes < 2000:
        return {
            "mensaje": f"La película '{title}' no cumple con el mínimo de 2000 valoraciones. "
                       f"Actualmente tiene {votes} votos registrados. El promedio de votación es {average:.1f}."
        }

    # Respuesta exitosa
    return {
        "titulo": title,
        "cantidad_de_votos": int(votes),
        "promedio_de_votacion": float(average),
    }
    
#Funcion que recibe el nombre de un actor
@app.get("/get_actor/{nombre_actor}")

def get_actor(nombre_actor: str):
    
    dataset = pd.read_csv("./dataset_actor_director.csv")    
    
    # Filtrar las filas donde el actor aparece en 'actor_names'
    actor_data = dataset[dataset['actor_names'].str.contains(nombre_actor, na=False, case=False)]

    if actor_data.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")

    # Calcular métricas
    cantidad_peliculas = actor_data.shape[0]
    retorno_total = actor_data['return'].sum()
    promedio_retorno = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0

    # Crear respuesta
    respuesta = {
        "mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_total:.2f} con un promedio de {promedio_retorno:.2f} por filmación."
    }
    return respuesta


# Funcion que recibe el nombre de un director
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    
    dataset = pd.read_csv("./dataset_actor_director.csv")
    
    # Filtrar las películas dirigidas por el director
    director_movies = dataset[dataset["director_name"].str.contains(nombre_director, case=False, na=False)]

    if director_movies.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")

    # Calcular el retorno total y construir la respuesta
    total_return = director_movies["return"].sum()
    movies_list = []

    for _, row in director_movies.iterrows():
        profit = row["revenue"] - row["budget"] if row["budget"] > 0 else 0

        movies_list.append({
            "titulo": row["title"],
            "fecha_lanzamiento": row["release_date"],
            "retorno_individual": row["return"],
            "presupuesto": row["budget"],
            "ganancia": profit
        })

    return {
        "nombre_del_director": nombre_director,
        "retorno_total": total_return,
        "peliculas": movies_list
    }
 
    
#SISTEMA DE RECOMENDACION
dataset = pd.read_csv("./dataset_modelo.csv")
# Asegurarnos de que las columnas estén correctamente tipificadas
dataset['vote_average'] = pd.to_numeric(dataset['vote_average'], errors='coerce')
dataset['vote_count'] = pd.to_numeric(dataset['vote_count'], errors='coerce')
dataset['popularity'] = pd.to_numeric(dataset['popularity'], errors='coerce')

# Convertir tipos de datos
dataset['vote_average'] = dataset['vote_average'].astype('float32')
dataset['vote_count'] = dataset['vote_count'].astype('int32')
dataset['popularity'] = dataset['popularity'].astype('float32')

# Criterios de filtro
vote_threshold = 100  # Mínimo de 100 votos
popularity_threshold = dataset['popularity'].quantile(0.9)  # Percentil 90 de popularidad

# Filtrar el dataset
filtered_movies = dataset[(dataset['vote_count'] >= vote_threshold) & (dataset['popularity'] >= popularity_threshold)].reset_index(drop=True)

# Preprocesar la columna 'overview'
filtered_movies['overview'] = filtered_movies['overview'].fillna('')

# Función segura para evaluar listas en 'genre_names'
def safe_literal_eval(x):
    try:
        return ' '.join(ast.literal_eval(x)) if isinstance(x, str) else ''
    except (ValueError, SyntaxError):
        return ''  # Devuelve una cadena vacía si no se puede evaluar

# Convertir 'genre_names' a una lista de géneros
filtered_movies['genre_names'] = filtered_movies['genre_names'].fillna('[]').apply(safe_literal_eval)

# Combinar 'overview' y 'genre_names' para enriquecer las descripciones
filtered_movies['combined_features'] = filtered_movies['overview'] + ' ' + filtered_movies['genre_names']
# Crear matriz TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered_movies['combined_features'])

# Calcular matriz de similitud del coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Crear mapeo de títulos a índices (normalizando a minúsculas)
indices = pd.Series(filtered_movies.index, index=filtered_movies['title'].str.lower()).drop_duplicates()

# Función para obtener recomendaciones
def get_recommendations(title, cosine_sim=cosine_sim):
    # Normalizar el título ingresado a minúsculas
    title = title.lower()
    
    if title not in indices:
        return ["Título no encontrado."]
    
    # Obtener el índice de la película
    idx = indices[title]
    
    # Calcular similitud
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Obtener los 5 más similares (excluyendo el propio)
    
    # Obtener los títulos de las películas recomendadas
    movie_indices = [i[0] for i in sim_scores]
    return filtered_movies['title'].iloc[movie_indices].tolist()


# Endpoint para la recomendación
@app.get("/recomendacion/")
def recomendacion(titulo: str):
    recomendaciones = get_recommendations(titulo)
    return {"recomendaciones": recomendaciones}
